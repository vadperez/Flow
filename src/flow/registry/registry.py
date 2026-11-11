"""Module registry: a SQLite index of module packs, their lifecycle status, events and usage.

The pack folders stay the source of truth for weights and manifests; the registry indexes them,
enforces the lifecycle (spec section 5) and keeps the history. One file, no server.

    python -m flow.registry scan library/          # index every pack under a folder
    python -m flow.registry list
    python -m flow.registry show fn.add_exact.v1
    python -m flow.registry status fn.add_exact.v1 active --reason "passed G2"
    python -m flow.registry verify                 # files still match their recorded hashes?
    python -m flow.registry export reports/registry.json   # for dashboards and reports
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import manifest as mf

SCHEMA = """
CREATE TABLE IF NOT EXISTS modules (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, function TEXT NOT NULL, domain TEXT, parent TEXT,
    version TEXT NOT NULL, status TEXT NOT NULL, fitted_to TEXT, portable_body INTEGER NOT NULL,
    params INTEGER, bytes INTEGER, path TEXT NOT NULL, sha256 TEXT, added TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS modules_status ON modules(status);
CREATE INDEX IF NOT EXISTS modules_function ON modules(function);
CREATE TABLE IF NOT EXISTS events (
    n INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT NOT NULL, module TEXT NOT NULL, event TEXT NOT NULL, detail TEXT
);
CREATE INDEX IF NOT EXISTS events_module ON events(module);
CREATE TABLE IF NOT EXISTS usage (
    module TEXT NOT NULL, day TEXT NOT NULL, calls INTEGER NOT NULL, PRIMARY KEY (module, day)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RegistryError(Exception):
    pass


class Registry:
    def __init__(self, path: "str | Path" = "registry/registry.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _event(self, module: str, event: str, detail: str = "") -> None:
        self.db.execute("INSERT INTO events (time, module, event, detail) VALUES (?, ?, ?, ?)", (now(), module, event, detail))

    # ---- writing ----------------------------------------------------------------------------------
    def add(self, folder: "str | Path") -> str:
        """Index one pack folder (or refresh it). The pack must be valid and its files must match their hashes."""
        folder = Path(folder)
        found = mf.pack_problems(folder)
        if found:
            raise RegistryError(f"{folder} is not a valid pack: " + "; ".join(found))
        m = mf.read(folder / "manifest.yaml")
        cost = m.get("cost") or {}
        known = self.db.execute("SELECT status FROM modules WHERE id = ?", (m["id"],)).fetchone()
        row = (m["id"], m["kind"], m["function"], m.get("domain"), m.get("parent"), m["version"], m["status"], m.get("fitted_to"),
               int(m["portable_body"]), cost.get("params"), cost.get("bytes"), str(folder), json.dumps(m.get("sha256")), now(), now())
        if known is None:
            self.db.execute("INSERT INTO modules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
            self._event(m["id"], "added", f"status {m['status']}, {cost.get('bytes', 0)} bytes")
        else:
            self.db.execute("UPDATE modules SET kind=?, function=?, domain=?, parent=?, version=?, status=?, fitted_to=?, portable_body=?, "
                            "params=?, bytes=?, path=?, sha256=?, updated=? WHERE id=?", row[1:13] + (now(), m["id"]))
            self._event(m["id"], "refreshed", "")
        self.db.commit()
        return m["id"]

    def scan(self, root: "str | Path") -> "tuple[list[str], dict[str, str]]":
        """Index every pack under `root`. Returns (ids added, problems by folder)."""
        added, failed = [], {}
        for manifest in sorted(Path(root).glob("*/manifest.yaml")):
            try:
                added.append(self.add(manifest.parent))
            except RegistryError as exc:
                failed[str(manifest.parent)] = str(exc)
        return added, failed

    def set_status(self, module: str, status: str, reason: str = "") -> None:
        """Move a module along its lifecycle; the manifest file is updated as well."""
        row = self.db.execute("SELECT status, path FROM modules WHERE id = ?", (module,)).fetchone()
        if row is None:
            raise RegistryError(f"unknown module {module}")
        if status not in mf.TRANSITIONS[row["status"]]:
            allowed = ", ".join(sorted(mf.TRANSITIONS[row["status"]])) or "nothing (final state)"
            raise RegistryError(f"{module} is '{row['status']}'; from there the lifecycle allows: {allowed}")
        path = Path(row["path"]) / "manifest.yaml"
        m = mf.read(path)
        m["status"] = status
        mf.write(m, path)
        self.db.execute("UPDATE modules SET status = ?, updated = ? WHERE id = ?", (status, now(), module))
        self._event(module, "status", f"{row['status']} -> {status}" + (f": {reason}" if reason else ""))
        self.db.commit()

    def record_usage(self, module: str, calls: int, day: "str | None" = None) -> None:
        """Add to a module's call count for a day (input to the suppression rule of spec section 5)."""
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.db.execute("INSERT INTO usage (module, day, calls) VALUES (?, ?, ?) "
                        "ON CONFLICT(module, day) DO UPDATE SET calls = calls + excluded.calls", (module, day, int(calls)))
        self.db.commit()

    # ---- reading ----------------------------------------------------------------------------------
    def list(self, status: "str | None" = None, kind: "str | None" = None) -> "list[dict]":
        query, args = "SELECT * FROM modules WHERE 1=1", []
        if status:
            query, args = query + " AND status = ?", args + [status]
        if kind:
            query, args = query + " AND kind = ?", args + [kind]
        return [dict(r) for r in self.db.execute(query + " ORDER BY id", args)]

    def show(self, module: str) -> dict:
        row = self.db.execute("SELECT * FROM modules WHERE id = ?", (module,)).fetchone()
        if row is None:
            raise RegistryError(f"unknown module {module}")
        events = [dict(r) for r in self.db.execute("SELECT time, event, detail FROM events WHERE module = ? ORDER BY n", (module,))]
        calls = self.db.execute("SELECT COALESCE(SUM(calls), 0) FROM usage WHERE module = ?", (module,)).fetchone()[0]
        return dict(row, events=events, calls_total=calls)

    def verify(self) -> "dict[str, list[str]]":
        """Problems per module: missing folders, changed files, manifests that no longer match the index."""
        found = {}
        for row in self.db.execute("SELECT id, path, status, sha256 FROM modules"):
            folder = Path(row["path"])
            issues = mf.pack_problems(folder) if folder.exists() else [f"folder {folder} is gone"]
            if not issues:
                m = mf.read(folder / "manifest.yaml")
                if m["status"] != row["status"]:
                    issues.append(f"manifest says '{m['status']}', registry says '{row['status']}'")
                if json.dumps(m.get("sha256")) != row["sha256"]:
                    issues.append("files changed since the pack was indexed (run: scan)")
            if issues:
                found[row["id"]] = issues
        return found

    def export(self, path: "str | Path", last_events: int = 200) -> dict:
        data = {"exported": now(), "modules": self.list(),
                "events": [dict(r) for r in self.db.execute("SELECT time, module, event, detail FROM events ORDER BY n DESC LIMIT ?", (last_events,))],
                "usage": [dict(r) for r in self.db.execute("SELECT module, SUM(calls) AS calls FROM usage GROUP BY module")],
                "by_status": {r["status"]: r["n"] for r in self.db.execute("SELECT status, COUNT(*) AS n FROM modules GROUP BY status")},
                "resident_bytes_if_all_loaded": self.db.execute("SELECT COALESCE(SUM(bytes), 0) FROM modules WHERE status = 'active'").fetchone()[0]}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
        return data

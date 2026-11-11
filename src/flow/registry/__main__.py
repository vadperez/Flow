"""Command line of the module registry. See `python -m flow.registry --help`."""
from __future__ import annotations

import argparse
import sys

from .registry import Registry, RegistryError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m flow.registry", description="Flow module registry")
    parser.add_argument("--db", default="registry/registry.sqlite", help="registry file (created if missing)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create an empty registry")
    p = sub.add_parser("add", help="index one or more pack folders")
    p.add_argument("folders", nargs="+")
    p = sub.add_parser("scan", help="index every pack under a library folder")
    p.add_argument("root")
    p = sub.add_parser("list", help="list modules")
    p.add_argument("--status")
    p.add_argument("--kind")
    p = sub.add_parser("show", help="one module with its history")
    p.add_argument("module")
    p = sub.add_parser("status", help="move a module along its lifecycle")
    p.add_argument("module")
    p.add_argument("new_status")
    p.add_argument("--reason", default="")
    sub.add_parser("verify", help="check that pack files still match their recorded hashes")
    p = sub.add_parser("export", help="write the registry as JSON (for dashboards and reports)")
    p.add_argument("file")
    args = parser.parse_args(argv)

    registry = Registry(args.db)
    try:
        if args.command == "init":
            print(f"registry ready: {registry.path}")
        elif args.command == "add":
            for folder in args.folders:
                print("indexed", registry.add(folder))
        elif args.command == "scan":
            added, failed = registry.scan(args.root)
            print(f"indexed {len(added)} pack(s): {', '.join(added) or '-'}")
            for folder, why in failed.items():
                print(f"NOT indexed {folder}: {why}")
            return 1 if failed else 0
        elif args.command == "list":
            rows = registry.list(args.status, args.kind)
            print(f"{'id':34} {'kind':12} {'status':15} {'bytes':>11}  fitted to")
            for r in rows:
                print(f"{r['id']:34} {r['kind']:12} {r['status']:15} {r['bytes'] or 0:>11,}  {r['fitted_to'] or '(body only)'}")
            print(f"{len(rows)} module(s)")
        elif args.command == "show":
            info = registry.show(args.module)
            for key, value in info.items():
                if key != "events":
                    print(f"{key:14} {value}")
            for e in info["events"]:
                print(f"  {e['time']}  {e['event']:10} {e['detail']}")
        elif args.command == "status":
            registry.set_status(args.module, args.new_status, args.reason)
            print(f"{args.module} is now {args.new_status}")
        elif args.command == "verify":
            found = registry.verify()
            for module, issues in found.items():
                print(f"{module}: " + "; ".join(issues))
            print("all packs match their records" if not found else f"{len(found)} module(s) with problems")
            return 1 if found else 0
        elif args.command == "export":
            data = registry.export(args.file)
            print(f"{len(data['modules'])} module(s) written to {args.file}")
    except RegistryError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

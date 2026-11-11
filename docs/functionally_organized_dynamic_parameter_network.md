# A functionally organized neural network with dynamic parameter memory

**Conceptual research note — 30 October 2026**  
**Status:** Proposed architecture and research direction, not an implemented model or engineering specification.

## Central idea

The proposal is to rethink neural-network architecture **before training**, rather than attempt to extract modular functions from a large, already-trained model. The objective is a **small standardized computational system** that has a persistent, potentially extensive library of learned parameters and **recruits only a small subset into active computation for each task**. The active parameter set can be replaced or recombined rapidly when the task changes.

This is not a proposal to keep a complete, conventional LLM in GPU memory, nor to make an ordinary network merely smaller. It is a proposal to make **functional organization and memory hierarchy first-class properties of the architecture**. Success would mean that total retained competence need not entail loading or executing all the learned parameters at once.

The model's permanent parameter library may be relatively large. **The particularly important reduction is in active parameters and active computation**, while reducing the total amount of independently learned parameters remains a desirable, separate research objective. No fixed numerical compression ratio is presumed achievable: reductions of several orders of magnitude are an aspiration to test, not a documented result.

## Why begin with function?

Contemporary neural networks are designed largely by specifying mathematical operations and their dimensions—layers, attention, gating, matrix widths, and so forth—then training their weights on broad objectives. Some functional specialization emerges, but detailed cognitive roles are not usually prescribed in advance. The proposed approach would reverse the emphasis: start by asking which reusable **functions, relationships, forms of integration, regulatory processes, and internal states** cognition requires, then determine what computational structures implement them.

An important source of candidate functions is mechanistic interpretability research on existing networks. Researchers have identified circuits associated with sequence copying and pattern completion (induction heads), selection and transfer of entity information (name-mover and indirect-object circuits), suppression of competing predictions, ordered transformations (successor heads), compact representation of task operations (function vectors), and limited compositional transformations. Work tracing larger models has also found examples of shared cross-linguistic concepts, intermediate relational reasoning, approximate and exact calculation pathways, and anticipatory planning. These are **partial, experimentally studied mechanisms**, not a complete taxonomy of cognition; a circuit can serve multiple functions, and a function may require distributed cooperation.

Thus, these observations should inform a **candidate functional vocabulary**, not be copied unquestioningly as the final architecture. The goal is to identify operations that can be composed and reused, including across domains, rather than to assign a separate large block of weights to every subject (biology, physics, law, etc.).

## Developmental organization, not rigid predestination

The biological premise is that neuronal populations can begin with **broad but type-constrained functional potential** and undergo differentiation, refinement, regulation, and selective elimination. Not every computational unit needs a completely predetermined final function. Instead, the architecture would specify what kinds of functional relationships and transformations are possible, then allow the useful organization to emerge within those constraints during centralized development and optimization.

**Elimination of excessive elements is a necessary developmental outcome.** Redundant connections, duplicate functions, unused structures, and unnecessary independent specifications should be consolidated, suppressed, or removed when doing so preserves required capabilities. Reversible suppression and irreversible elimination must be distinguished: a rarely used function may still be indispensable under unfamiliar circumstances. Some broad, reusable functions should remain broad instead of being forced to specialize.

A related goal is replacing large numbers of separately specified connections with **shared functional or regulatory rules**, where feasible. A compact rule may determine the behavior of many interactions. This is a hypothesis about representational economy—not an assumption that all knowledge or computation can be reduced to a tiny rulebook.

## Permanent memory and active working set

The desired physical hierarchy is:

- **Nonvolatile permanent parameter memory (SSD or future on-chip/nonvolatile storage):** the validated library of learned parameter sets, potentially much larger than the active model.
- **Intermediate system memory (RAM):** a cache of frequently needed or anticipated parameter sets, facilitating rapid replacement without constant SSD access.
- **Active parameter memory (GPU VRAM or a future neuromorphic processor):** the small computational core, function-selection/control machinery, and only the parameter groups needed for the current work.
- **Transient computational state:** the immediately evolving information used to process a task. This is distinct from the GPU-resident parameter subset; in neuroscience and machine learning, *working memory* ordinarily refers more specifically to temporary state, rather than simply the current selection of long-term weights.

A small controller would interpret the requirements of a task, identify a viable combination of functional modules, resolve their dependencies, and arrange the loading or reuse of their parameter sets. The active configuration should persist while useful and change when the problem changes. Transfers should generally be batched around meaningful operations or task boundaries, not performed for arbitrary weights on every token. Unused learned parameters need not occupy GPU memory or participate in inference.

This scheme requires **genuine executable modularity**. A common numerical input/output size alone does not guarantee that independently developed modules exchange compatible representations. Their functional contracts, composition rules, and shared representation conventions must be developed and validated. The design should also permit combinations of modules, since complex tasks often require several interdependent operations rather than one topical specialist.

## What the proposal is—and is not

It is **function-first**, rather than a dense network retroactively pruned by topic. It is **memory-hierarchical by design**, rather than merely offloading randomly chosen conventional layers. It allows **large persistent knowledge capacity but small active computation**, without requiring all stored parameters to become active for every input. It favors **functional reuse, differentiation, integration, and selective elimination**. It does **not** assume that every neuron, concept, or domain has an exclusive assignment. It does **not** treat parameter count alone as a sufficient measure of efficiency: transfer latency, state size, energy, computation, and any external storage must also be counted.

The architecture is envisioned as a **standardized product**. Its functions and validated learned parameter library could be developed centrally and distributed identically. **Individual lifetime learning is optional**, not a prerequisite of the main model. A robotics version could acquire additional experience-specific adaptations while retaining the standardized baseline. Useful optional learning must not be confused with the process of training and validating the original product.

## A sensible research progression

The initial untrained executable framework could be small. Creating such a framework is comparatively straightforward; establishing that its functional primitives and coordination mechanisms can acquire significant competence is the real scientific challenge. External training sets can provide learning objectives, and **separate unseen tests** must evaluate competence and generalization. Evaluation by itself does not train a network unless an optimization or selection mechanism uses the results as feedback.

A first demonstration need not start with unrestricted natural language. It could test a small number of independently recognizable transformations, their composition in new combinations, removal of redundant structures, and the ability to switch parameter sets without loss of competence. A conventional small network should serve as a resource-matched control. Additional capabilities, including language understanding, could follow once the functional and memory principles have been established.

The central experimental measures should be: **capability per active parameter; capability per total retained parameter; generalization to new functional combinations; competence after elimination; module-switching cost; memory traffic; energy; and functional reliability under unfamiliar tasks**. The decisive result would be increasingly broad capability without proportionate growth in the active network. The stronger goal—several-orders-of-magnitude improvement while retaining broad general intelligence—remains unproven.

An experimental software environment should make the architecture inspectable and reproducible: active-function and parameter-residency monitoring, module and checkpoint management, training/evaluation comparisons, safe rollback, loading traces, resource measurement, and import/export of datasets, parameter libraries, and results. These are development tools rather than defining properties of the cognitive architecture.

## Neuromorphic hardware as an eventual outcome

If a small, reusable functional repertoire is found, **the active inference machinery could eventually be embodied directly in silicon**. The chip would implement the operations, regulatory states, local memory, selective parameter loading, and interactions that the validated architecture actually needs—not simply reproduce large matrix-multiplication hardware at smaller scale. This implies a progression from software experiments to a reconfigurable implementation (possibly an FPGA) and, only after stable results, to application-specific silicon.

The hardware objective should not determine the functional taxonomy prematurely. Nor should a lower learned parameter count conceal the cost of hardwired circuitry, retained external information, or repeated computation.

## Open questions

The largest scientific unknown is **whether a small repertoire of functionally organized mechanisms can support high general competence while activating only a tiny subset of a potentially large permanent parameter library**. Other unresolved questions include: which functions are genuinely primitive; which must be shared; how interfaces remain compatible after differentiation; when suppression becomes safe elimination; how a controller selects parameter sets without performing most of the target computation itself; and whether the memory-transfer cost is sufficiently low to make dynamic configuration advantageous.

No existing result establishes that this architecture can reproduce a frontier LLM with millions of active parameters. Conversely, parameter scaling observed in contemporary dense or mixture-of-experts language models is not proof that such a function-first architecture is impossible.

## Relationship to current frontier models

As of October 2026, the detailed neural architectures and parameter counts of **Claude Fable 5.1 / Mythos 5.1** and **GPT-6 Astra** are not publicly specified in a way that establishes whether they employ comparable functional modularity or memory hierarchy. Anthropic states that Fable 5.1 and Mythos 5.1 are the **same underlying model**, differentiated by safeguards and access conditions. It would be premature to describe either company's frontier model as an implementation of this proposal, or to insist that its internal design is simply a conventional dense transformer.

## Working statement of the hypothesis

> A standardized, functionally organized neural architecture can retain a large repertoire of learned competencies in permanent parameter memory while composing a much smaller set of active computational parameters for each task. Developmental differentiation, functional reuse, regulation, and selective elimination may improve competence per independent parameter. The resulting active inference mechanism could ultimately be implemented in specialized neuromorphic hardware, while individual experience-dependent learning remains an optional extension.

## Selected references

These works support particular mechanisms or background observations; none proves the complete proposal.

1. Olsson et al. (2022), *In-context Learning and Induction Heads*: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
2. Wang et al. (2023), *Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small*: https://arxiv.org/abs/2211.00593
3. McDougall et al. (2024), *Copy Suppression: Comprehensively Understanding a Motif in Language Model Attention Heads*: https://aclanthology.org/2024.blackboxnlp-1.22/
4. Gould et al. (2024), *Successor Heads: Recurring, Interpretable Attention Heads in the Wild*: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2722a0ccf6acfe3d144fdbb0dedd80b5-Abstract-Conference.html
5. Todd et al. (2024), *Function Vectors in Large Language Models*: https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ae163cb8788970e53b4fd9578141139-Abstract-Conference.html
6. Anthropic (2025), *Tracing the thoughts of a large language model*: https://www.anthropic.com/research/tracing-thoughts-language-model
7. Tang, Lake & Jazayeri (2026), *Circuit explained: How does a transformer perform compositional generalization*: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0340088
8. Anthropic (2026), *Introducing Claude Fable 5.1 and Claude Mythos 5.1*: https://www.anthropic.com/claude-fable-and-mythos-5-1
9. Anthropic (2026), *Transparency Hub*: https://www.anthropic.com/transparency
10. OpenAI (2026), *GPT-6 Astra*: https://openai.com/index/gpt-6-astra/
11. OpenAI, *GPT-6 Astra API model page*: https://developers.openai.com/api/docs/models/gpt-6-astra

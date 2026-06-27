# Scripts Layout

The scripts are grouped by the part of the submission pipeline they support.

- `graphs/`: attribution graph generation and the local graph-viewer wrapper.
- `features/`: top-K activation and feature-labelling SLURM wrappers.
- `interventions/`: supernode sweeps, random baselines, top-logit sweeps, and steerable-ceiling diagnostics.
- `evaluation/`: replacement-model fidelity and CSD3 test-suite wrappers.
- `refusal/`: base-vs-Heretic refusal graph and cross-model feature comparison jobs.
- `figures/`: plotting helpers that consume saved JSON outputs.

"""Feature preprocessing pipelines that produce inputs for novel architectures.

Each submodule produces deterministic, cacheable features:
  - topology: persistent-homology embeddings of rolling correlation matrices
  - causal:   regime-conditioned DAGs from NOTEARS (Option A)

Caches live in /scratch/users/$USER/ra-tft/features/{topology,causal}/ on the cluster,
or in features/cache/ locally.
"""

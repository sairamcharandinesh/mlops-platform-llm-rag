# Overview

This project implements an MLOps-style platform for RAG:

- **Orchestration** — FastAPI with ingest and query endpoints
- **Models** — BentoML for embeddings and generation
- **Data** — LakeFS versioning, Qdrant vectors, MongoDB for evaluation data
- **Observability** — Prometheus + Grafana
- **Tracking** — MLflow for experiments and evaluation runs

All services are containerised and orchestrated with Docker Compose. See the main README for setup and URLs.

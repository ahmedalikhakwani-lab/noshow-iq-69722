# NoShowIQ

[![CI/CD](https://github.com/ahmedalikhakwani-lab/noshow-iq-69722/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ahmedalikhakwani-lab/noshow-iq-69722/actions/workflows/ci-cd.yml)

Predict whether a patient will miss their clinic appointment.
Built for MLOps Midterm — Spring 2026, Riphah International University.

**Live URL:** https://khakwani16-noshow-iq.hf.space  
**Docker Hub:** https://hub.docker.com/r/YOUR_DOCKERHUB_USERNAME/noshow-iq  
**TestPyPI:** https://test.pypi.org/project/noshow-iq-69722/

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/predict` | Predict no-show risk |
| GET | `/history` | Last 20 predictions |
| GET | `/stats` | Aggregated MongoDB stats |

---

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
python -m uvicorn noshow_iq.api:app --reload
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## Smoke Test

```bash
python smoke_test.py https://khakwani16-noshow-iq.hf.space
```
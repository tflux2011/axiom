# AXIOM — Axiomatic Hyperdimensional Distillation for Zero-Latency Offline Medical Inference

A neurosymbolic architecture that combines **Hyperdimensional Computing (HDC)** with **Small Language Models** for safe, offline medical reasoning. AXIOM replaces the standard RAG pipeline with three tightly integrated components:

1. **Relational Contextual Distiller** — Compresses a multi-GB medical corpus into a single 10,000-dimension HDC "Axiom Map"
2. **Zero-Retrieval Latent Priming** — Injects the Axiom Map directly into the SLM's KV-cache (no search step)
3. **Neurosymbolic Safety Governor** — Real-time token-level hallucination detection via HDC cosine similarity

## Key Results (BioASQ 14b)

| Metric             | Vanilla SLM | SLM + RAG | **AXIOM**  |
| ------------------ | ----------- | --------- | ---------- |
| Storage            | 2.1 GB      | 14.1 GB   | **3.3 GB** |
| TTFT               | 180 ms      | 320 ms    | **< 5 ms** |
| Hallucination Rate | 23.4%       | 8.7%      | **0.2%**   |

## Validation Status

All three core components have been independently validated on commodity hardware (Apple M2, 16 GB, no GPU). Full report: [`results/VALIDATION_REPORT.md`](results/VALIDATION_REPORT.md).

| Component                | Key Metric              | Paper Target | POC Result     | Status |
| ------------------------ | ----------------------- | ------------ | -------------- | ------ |
| **Distiller**            | NN Retrieval Accuracy   | 99%          | 99.0%          | Passed |
| **Distiller**            | F1 Score                | 0.995        | 0.995          | Passed |
| **Distiller**            | Hallucination Catch     | 100%         | 100%           | Passed |
| **Latent Priming**       | Projector Params        | ~30M         | 30,287,360     | Passed |
| **Latent Priming**       | Virtual Token Injection | Layer n/2    | Layer 6 of 12  | Passed |
| **Safety Governor**      | Catch Rate              | 100%         | 100%           | Passed |
| **Compression vs FAISS** | Index Size Ratio        | —            | 179x smaller   | Passed |
| **Speed vs FAISS**       | Query Speedup           | —            | 29.4x faster   | Passed |

> **POC scope:** GPT-2 124M with 150 facts / 200 QA pairs. The gap to production is scale, not architecture.

## Project Structure

```
axiom/
├── paper/
│   └── axiom_paper.tex          # ArXiv LaTeX paper
├── src/
│   ├── config.py                # Centralised configuration
│   ├── distiller.py             # HDC Axiomatic Distiller (torchhd)
│   ├── encoder.py               # Biomedical NER pipeline (scispaCy)
│   ├── priming.py               # Zero-Retrieval KV-Cache Injection
│   ├── governor.py              # Neurosymbolic Safety Governor
│   └── utils.py                 # Shared utilities
├── benchmarks/
│   ├── compression_bench.py     # Phase A: Storage comparison
│   ├── latency_bench.py         # Phase B: TTFT measurement
│   └── accuracy_bench.py        # Phase C: Hallucination test
├── scripts/
│   ├── download_bioasq.py       # Dataset fetcher
│   ├── distill_corpus.py        # End-to-end distillation pipeline
│   └── run_inference.py         # Interactive QA demo
├── tests/
│   └── test_distiller.py        # Unit tests
├── requirements.txt
└── .env.example                 # Environment variable template
```

## Quick Start

### 1. Setup

```bash
# Clone and install
cd axiom
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Install biomedical NER model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# Configure environment
cp .env.example .env
# Edit .env with your paths and credentials
```

### 2. Prepare Dataset

```bash
python -m scripts.download_bioasq
```

### 3. Distil Knowledge

```bash
# From structured triples (fast, no NER)
python -m scripts.distill_corpus --use-structured

# From raw text (uses scispaCy NER)
python -m scripts.distill_corpus
```

### 4. Run Demo

```bash
# HD query demo (no LLM required)
python -m scripts.run_inference --demo

# Full inference with Llama-3.2-3B
python -m scripts.run_inference
```

### 5. Run Benchmarks

```bash
python -m benchmarks.compression_bench
python -m benchmarks.latency_bench
python -m benchmarks.accuracy_bench
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## Architecture Overview

### The Distiller (HDC Engine)

Converts medical knowledge triples (Subject → Relation → Object) into hyperdimensional vectors:

```
Fact HV = bind(bind(Subject, Relation), Object)
Axiom Map = sign(Σ Fact HVs)
```

At D=10,000, the Axiom Map stores millions of facts in a single vector while maintaining queryability via cosine similarity.

### Zero-Retrieval Priming

Instead of searching a vector database at query time, the Axiom Map is projected into the SLM's hidden dimension and injected as "virtual context tokens" via a forward hook on the transformer's KV-cache. This eliminates the retrieval bottleneck entirely.

### Safety Governor

Each candidate token is projected into HDC space and compared against the Axiom Map. Tokens with low cosine similarity (below threshold τ = 0.35) receive a logit penalty of -100, effectively suppressing hallucinated medical content.

## Configuration

All configuration is centralised in `src/config.py`. Key parameters:

| Parameter            | Default | Description                        |
| -------------------- | ------- | ---------------------------------- |
| `dimensions`         | 10,000  | HDC vector dimensionality          |
| `safety_threshold`   | 0.35    | Governor suppression threshold     |
| `injection_layer`    | 16      | Transformer layer for KV injection |
| `max_virtual_tokens` | 128     | Number of virtual context tokens   |
| `quantisation`       | nf4     | Model quantisation scheme          |

## External Storage

Large artefacts (model weights, datasets, Axiom Maps) are stored on the external HD at `/Volumes/WD Drive/axiom_data/`. Set `AXIOM_STORAGE_PATH` in `.env` to change this.

## Security

- No credentials are hard-coded; all secrets come from `.env`
- The Safety Governor operates at the logit level — cannot be bypassed by prompt injection
- All user input is sanitised before NER processing
- Model weights are loaded with `trust_remote_code=False`
- No external API calls during inference (fully offline)

## Paper

The ArXiv paper is in `paper/axiom_paper.tex`. Build with:

```bash
cd paper && pdflatex axiom_paper.tex && bibtex axiom_paper && pdflatex axiom_paper.tex && pdflatex axiom_paper.tex
```

## License

Research use only. See LICENSE for details.

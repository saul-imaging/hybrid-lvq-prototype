# Hybrid LVQ Prototype

Prototype implementation of a Hybrid Learning Vector Quantization (HLVQ) classifier for tabular and biomedical-style datasets.

The project combines online prototype updates, batch refinement from top-k assigned samples, stochastic exploration through a Levy-based learning-rate schedule, prototype dropout, and reseeding of inactive prototypes.

## Highlights

- Implements a reusable `train_hybrid_lvq` function with reproducible seeds and validation tracking.
- Uses the OpenML Splice dataset as a runnable DNA sequence classification example.
- Includes selected result figures for previous benchmark runs.

## Repository Structure

```text
.
├── hybrid_lvq.py
├── run_splice_hlvq.py
├── requirements.txt
└── results/
    ├── banknote/
    └── splice/
```

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_splice_hlvq.py
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_splice_hlvq.py
```

## Notes

- `run_splice_hlvq.py` downloads the Splice dataset from OpenML at runtime.
- The selected result images are included only as reference outputs; raw datasets are not versioned.


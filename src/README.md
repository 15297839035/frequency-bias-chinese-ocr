# Code

Research code used for the experiments in the paper, grouped by role. It is the original
working code, cleaned up and renamed; it is not packaged or installed as a library.

| Directory | Contents |
|---|---|
| `core/` | Shared correction and selection logic (`prior_correction.py`), self-test |
| `data/` | HWDB conversion and split construction; text-line swap preparation |
| `inference/` | CTC posterior dumps, external/industrial model inference, SMTR decoding |
| `training/` | HWDB classifiers, BCTR fine-tuning, CRNN training (plain, LA, line swap) |
| `evaluation/` | Re-evaluation and metric aggregation for HWDB and BCTR |
| `analysis/` | Confusion direction, interventions, ablations, τ curves, reports |
| `figures/` | Paper and revision figures |
| `pipelines/` | Original shell pipelines used on the GPU server |

See [`docs/experiments.md`](../docs/experiments.md) for the experiment → code → result →
paper-section map, including the mapping from the original `ly/scripts/...` names.

## Import path

Scripts insert `src/` on `sys.path` when run directly
(`Path(__file__).resolve().parents[1]`) and import the other modules as
`from core import prior_correction`, `from analysis.c03_confusion_analysis import ...`, etc.
When running a module as a script, either use `python src/<package>/<script>.py` or set:

```bash
export PYTHONPATH="$PWD/src"
```

The repository root is `Path(__file__).resolve().parents[2]` in every script, and data and
results are read from `data/` and `results/` at that root.

## Environments

* HWDB and analysis: Python 3.12, PyTorch 2.8, NumPy 1.26 (`numpy<2`).
* OpenOCR experiments: OpenOCR installed with `OpenOCR-main/` on `PYTHONPATH`; see the
  repository README.

Shell pipelines contain absolute paths from the authors' GPU server
(`/root/autodl-tmp/...`); edit the variables at the top before running.

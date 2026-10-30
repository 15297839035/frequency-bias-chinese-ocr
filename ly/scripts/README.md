# Code

Research code used for the experiments in the paper. The scripts are the original
working versions; they are not packaged or installed as a library.

* `a0_*`, `a1_*`: posterior dumps, look-alike/prior correction, HWDB preparation and training.
* `c1_*` … `c17_*`: the C-series experiments (evaluation, interventions, baselines, curves).
* `c4*`, `c5*`, `c9*`, `c11*`, `c16*`, `v4*`: OpenOCR-based text-recognition experiments.
* `m10_*`, `m11_*`, `*_pipeline.sh`, `*_cpu.sh`: orchestration and restart scripts.
* `w1_figures.py`, `v4_figures.py`: paper figures.

See [`docs/experiments.md`](../../docs/experiments.md) for the experiment → script → result
→ paper-section map.

## Paths

Python scripts compute `REPO = Path(__file__).resolve().parents[2]`, so keep the
`ly/scripts/` layout (`<repo>/ly/scripts/<script>.py`). Shell pipelines contain absolute
paths from the authors' server (`/root/autodl-tmp/...`); edit the variables at the top of
the script before running.

## Environments

* HWDB and analysis: Python 3.12, PyTorch 2.8, NumPy 1.26 (`numpy<2`).
* OpenOCR experiments: OpenOCR installed with `OpenOCR-main/` on `PYTHONPATH`; see the
  repository README.

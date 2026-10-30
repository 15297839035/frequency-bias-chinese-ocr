# Results

Aggregate outputs of the experiments: metric JSON files, `report.md` summaries, model
SHA-256 records, and small derived arrays (glyph-neighbour lists, split weights). The
directory names match the experiment IDs used in the paper and in
[`docs/experiments.md`](../../docs/experiments.md).

Not included:

* per-sample predictions (`dumps/`) and logits — several hundred MB, archived separately;
* model checkpoints;
* dataset images and annotations.

The JSON files may contain absolute paths from the authors' server (for provenance); they
are historical records and are not needed to read the metrics.

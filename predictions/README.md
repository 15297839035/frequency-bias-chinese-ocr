# Per-sample predictions

`per_sample_predictions.tar.gz` contains the per-sample prediction outputs (JSONL) of the
experiments in the paper.

- Files: 245 JSONL
- Uncompressed: 2,410,320,768 bytes
- Archive: 323,090,535 bytes
- SHA-256: `7ccac4675605f20f245057c5273aff4f97066d87901add05ee7d35b97771ab68`
- Stored with Git LFS.

The archive preserves repository-relative paths under `results/<experiment>/dumps/`. From the
repository root:

```bash
git lfs pull
tar -xzf predictions/per_sample_predictions.tar.gz -C .
```

This restores files such as `results/C-02_bctr_la/dumps/la025_s2_val_scene.jsonl`. The archive
contains no images, model weights, training logs or split indices; those are either in
`data/` (splits) or not distributed (images, weights).

If you do not use Git LFS, download the file directly from
<https://media.githubusercontent.com/media/heyuuuuuuuuu/frequency-bias-chinese-ocr/main/predictions/per_sample_predictions.tar.gz>
or use the download button on the GitHub file page.

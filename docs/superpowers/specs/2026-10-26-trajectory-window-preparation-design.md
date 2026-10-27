# Fixed-split trajectory window preparation

The user approved the next preparation step: validate target/history/future coverage using the frozen ancestry split and fit a normalizer from train only. The server has 951 complete recorded episodes, no rejected/excluded/duplicate files, and nonempty ancestry-disjoint splits. The complete server manifest and pool are unavailable locally; real local validation uses the uploaded single episode.

Use the existing canonical NumPy data boundary and add an independent CLI. A streaming `iter_windows` shares the old `make_windows` implementation; `make_windows` remains a list-returning compatibility wrapper. Neither replay nor training entry points change.

Defaults (implementation choices): history 11, future 30, dt 0.2, stride 1. History includes the anchor, spanning 2 seconds; future spans the following 6 seconds. Retain existing missing-sample and maximum-gap interpolation semantics. Require finite, masked ego history and at least one NPC with a fully valid history and future. Check all NPCs visible at the anchor; report contiguous history/future lengths and full-coverage counts. Report ego future completeness separately; never require or fit on future ego as a conditioning feature.

Use the ego anchor's fixed position/yaw to transform positions, velocities and yaw into ego-centric coordinates. Retain the eight existing fields and recorded position reference. Future NPC samples are supervised labels. The normalizer pools ego history and all fully covered target NPC history/future from eligible train windows only. Overlapping windows intentionally contribute repeatedly, matching the indexed sampling population. Use streaming moments; fit circular yaw mean then wrapped-residual standard deviation in a second train-only pass. Persist through the existing `ScenarioNormalizer` with coordinate_frame `ego-centric`; it cannot be substituted for a world-coordinate normalizer.

## Approaches

Reusing the old dataset CLI would re-split data, retain full pools/windows, and fit in world coordinates. Materializing training tensors would add storage and model-specific padding choices outside the approved preparation step. The chosen implementation is a new bounded-memory audit and index CLI with a small compatibility refactor, deriving each window from original NPZ files. The user authorized the preparation workflow; these routine implementation choices are stated in commentary and do not require another approval cycle.

## Artifact and trust contracts

Consume the existing `split_manifest.json`, preserving every membership. Recompute ancestry overlap and reject duplicate IDs or a manifest with a failed/incomplete preparation status. Resolve original relative paths with an optional relocated input root. Verify each recorded SHA256 and loaded identity, then verify files remain unchanged. Original NPZ files and the frozen manifest are read-only.

Write a new directory containing `window_index.jsonl` (one row per eligible window with split, source/hash, window ordinal, anchor and complete target IDs), `window_dataset_manifest.json`, `window_report.json`, and `normalizer.json` only after all checks succeed. No tensor NPZs or model training. Report per-split and per-episode coverage, exclusion reasons, manifest/source identities, fitting population and source provenance. Output directories must be new/empty.

Exit 0: all positively requested splits have eligible windows and the train normalizer is fitted. Exit 2: clean inputs but an eligible split is empty; preserve the report/index and do not emit a usable normalizer. Exit 1: input/identity/hash/manifest errors; preserve diagnostic partial outputs for per-episode errors, with no usable normalizer. Do not use partial outputs for training. Report provenance limitations rather than treating complete masks as simulator or physical validity.

## Validation

Test frozen memberships, transitive leakage rejection, hash/identity mismatches and relocation; strict missing-mask behavior and contiguous coverage; train-only normalization with extreme held-out data; circular yaw, local-coordinate moments and inverse conversion; empty/short/no-target data; input immutability and output protection; episode lifetime; second-pass changes; CLI progress/exit codes. Compare legacy `make_windows` outputs exactly against a before snapshot. Run the complete unittest suite and actual uploaded NPZ, then package a patch and server commands. No full-server counts are inferred locally.

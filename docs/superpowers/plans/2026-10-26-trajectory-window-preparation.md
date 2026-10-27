# Fixed-split trajectory window preparation implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation, followed by independent review.

**Goal:** Check complete NPC supervision windows under the frozen split and fit an auditable train-only ego-centric normalizer.

**Architecture:** Expose a window iterator without changing the list API; add a standalone manifest-driven scanner and two-pass streaming statistics; write indexes and reports rather than tensors.

**Tech Stack:** Python >=3.8, NumPy, stdlib; unittest. Local interpreter `D:/anaconda3/python.exe`.

**Spec:** `docs/superpowers/specs/2026-10-26-trajectory-window-preparation-design.md`.

## Global constraints

- Preserve fixed split membership and source files; no replay/training/checkpoint changes.
- Default history=11, future=30, dt=0.2, stride=1, ego-centric coordinates.
- Only eligible train windows fit statistics; no future ego inputs, no held-out fitting.
- No full-pool or full-episode-window array retention; no tensor materialization.
- Report errors/empty eligible splits without silently presenting a successful normalizer.
- No git checkout exists; retain a before snapshot and deliver a checked patch.

## Review focus

- A forged audit flag cannot permit transitive parent ancestry across splits.
- Changed source files, wrong loaded IDs and relocated roots must be handled explicitly.
- Missing NPC samples must not be imputed across gaps or normalized as zero.
- Held-out states must not affect normalizer statistics, including circular yaw.
- Eligibility may eliminate all windows in a populated split; report this as incomplete preparation.

### Task 1: Stream existing windows

Files: `embodied/envs/scenario_data.py`, `tests/test_trajectory_windows.py`.

- [x] Add a failing iterator/lifetime regression with hand-checked timestamps/masks.
- [x] Extract `iter_windows` from the current function and keep `make_windows` list behavior.
- [x] Run focused tests and compare existing output against the saved original.

### Task 2: Frozen-manifest window audit and normalizer

Files: new `embodied/envs/prepare_trajectory_windows.py`, same new test file.

API: `prepare_windows(manifest_path, output, history=11, future=30, dt=.2, stride=1, input_root=None, progress=None) -> dict`.

- [x] Write and run failing behavioral tests for the review focus plus CLI, coverage, output safety and episode release.
- [x] Implement manifest validation, streaming per-episode coverage/indexing and train-only moments.
- [x] Fit wrapped yaw residual variance in a second train-only pass and emit the existing normalizer schema with local-coordinate provenance.
- [x] Run focused and complete tests; fix demonstrated failures.

### Task 3: Real-data evidence, review and delivery

Files: `docs/trajectory_window_preparation.md`, `docs/window_preparation/*`, `artifacts/trajectory_window_preparation.*`.

- [x] Validate multi-group fixtures and the actual uploaded NPZ, comparing original hashes.
- [x] Independent review and regressions for any findings.
- [x] Document exact server manifest path, parameters, output interpretation and source limitations.
- [x] Build patch/ZIP; verify dry-run, application to originals, resulting files and archive CRC.

The user approved this preparation step. Continue inline with the established defaults and deliver runnable server commands; the full server pool remains inaccessible locally.

## Execution record

Pre-flight: Task 2 consumes the iterator from Task 1 with the identical window signature; Task 3 packages those same four production/test/document files against the original snapshot. No conflicting interfaces found. The checkout has no git metadata; original source is retained under `docs/window_preparation/source_before` rather than inventing branch/commit state.

Task 1: complete. Iterator regression was RED then GREEN. Independently compared all fields of 136 original/new windows across world/local frames, 0.1/0.2 dt, stride 1/2, events and timestamp gaps; exact equality including NaNs.

Task 2: complete. 17 initial regressions were RED then GREEN. Original suite plus additions passed 154 tests. Streamed statistics match the hand-derived local x mean 5.2 and 144 fitting states in the 8-group format fixture; extreme held-out states have no statistical effect.

Final review: independent read-only review identified one Important issue: a source changed after its last processing pass could evade final publication checks. Fixed with an all-source SHA256 sweep before publishing. `test_late_held_out_or_train_changes_prevent_publishing_success` failed for both held-out-between-passes and train-after-final-fit mutations, then passed. Final complete suite: `D:/anaconda3/python.exe -m unittest discover -s tests`, 155 tests, 17.575 seconds, OK. No remaining deferred minors.

Review limitations are retained as validation limits: the full 951-episode server pool is inaccessible locally; physical/simulator validity is outside mask/finite/identity checks. Real-data validation ran separately through the final CLI: uploaded episode gives 553 windows, 552 eligible, 5337 complete target trajectories and 224889 window-weighted fitting states; all 552 normalization round trips and original-source/manifest hashes passed. This is a single-episode train-only smoke check, not independent held-out validation.

Task 3: complete. Instructions include the user's actual server root and frozen manifest. `docs/window_preparation/verification.json`, `delivery_validation.json` and `package_checks.json` hold evidence. Patch dry-run, application to originals, four resulting-file comparisons and ZIP CRC all passed. Artifacts: `artifacts/trajectory_window_preparation.zip` and `.patch`. No source pool, split membership, checkpoint, training entry point or model was modified.

# Recorded episode inventory and parent-group split preparation

**Goal:** Count recorded episodes and independent recorded ancestry groups, then save deterministic train/val/test manifests without retaining an entire pool's arrays or materializing windows.

**Scope:** The user explicitly approved these two preparation steps after the Stage 3 server verification. Default ratios remain 0.8/0.1/0.1 and seed 0. This is not Diffusion implementation, normalization, or training.

**Architecture:** Extract the existing ancestry-component helper from `split_episodes` without changing assignments. Add an independent NumPy/stdlib CLI that loads and validates one NPZ at a time through `load_episode`, retains only a dataclass index, and calls the same grouping/splitting functions. It writes a frozen index, split manifest and report to a new output directory, with progress on stderr.

**Files:**
- Modify `embodied/envs/scenario_data.py`: expose `group_episodes`; retain exact legacy split ordering and ratios.
- Create `embodied/envs/prepare_scenario_splits.py`: streaming scan, recorded-source classification, duplicate/conflict handling, deterministic grouping, leakage audit and CLI.
- Create `tests/test_scenario_splits.py`: real NPZ fixtures and CLI/regression coverage.
- Create `docs/scenario_split_preparation.md`: commands, report meaning, incomplete-split behavior and limitations.

**Compatibility:** Existing dataset and training entry points keep their defaults. No original NPZ is rewritten. No windows or world-coordinate normalizer are created. Full scene/parent ancestry components stay together. 'Independent' refers to recorded lineage components, not proven semantic/map independence.

**Validation level:** Reuse canonical episode loading for schema, array-shape and timestamp checks. This does not prove per-window target availability or simulator validity. Accept recorded SUMO source types; report excluded synthetic/generated/unknown sources separately. Include executed incomplete prefixes with their completeness flag rather than silently deleting failures.

**Review focus:**
- Nested parent chains, absent parent records and repeated scenario IDs must stay in one split.
- Byte-identical duplicate episode IDs count once; differing files with the same ID must not silently select a winner.
- Input arrays/raw frames must be released before the next episode load.
- Empty requested splits must be reported without breaking ancestry groups.
- Generated candidates must not become recorded supervision, and new output directories must prevent silent overwrites.

- [x] Write failing behavioral tests for the public tool and shared grouping helper.
- [x] Implement the minimal streaming index/manifest pipeline and progress.
- [x] Run focused and full regression tests; compare legacy split assignments.
- [x] Validate a multi-source fixture pool and the uploaded real NPZ, keeping original hashes unchanged.
- [x] Review, document and package a server patch plus executable commands.

## Environment and execution notes

The supplied workspace has no `.git`; changes remain in the workspace with before snapshots and a reviewable patch. Work is executed inline within the user's approved scope. The server's full pool is not locally available; local real-data validation uses the uploaded episode and cannot invent the server's pool size or split distribution.

## Results

All 137 tests passed (13.603 seconds); 13 new tests include malformed root/nested metadata and damaged DEFLATE archives, added RED-to-GREEN after independent review. Final review found no remaining blockers. The full-episode lifetime regression confirms the previous episode is released before the next canonical load. Another 640 comparisons preserve legacy split membership/order exactly.

Generated format fixtures: 60 episodes, 20 ancestry groups, train 48/16, val 6/2, test 6/2 (episodes/groups), no leakage. These are test fixtures, not SUMO execution statistics. Uploaded real data: 1 episode, 1 group, train 1, val/test 0; correct exit 2 and unchanged source hash. Full server pool statistics remain a server-run prerequisite, not an inferred local result.

Evidence: `docs/dataset_split_preparation/verification.json`, `package_checks.json`, real-sample manifests, and the original-source snapshot. Server instructions: `docs/scenario_split_preparation.md`. Patch ZIP and patch are under `artifacts/scenario_split_preparation.*`; dry-run, snapshot application, all four resulting-file comparisons and ZIP CRC passed.

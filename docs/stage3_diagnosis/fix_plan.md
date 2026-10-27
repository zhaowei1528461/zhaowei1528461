# Stage 3 reference heading correction

**Goal:** Restore usable real-sample search without changing physical limits, risk definitions, recorded data, RSSM, or training.

**Evidence:** The uploaded episode and cfg/net/routes hashes match the local files. Replaying all 3072 original trials yields zero valid candidates. The selected NPC heading differs from the ego heading by 4.8–41.5 degrees. Holding a vehicle on the ego heading forces artificial lateral corrections even for `keep_lane`. A diagnostic experiment using the target heading, with unchanged evaluator and bounds construction, finds valid candidates in all six windows.

**Decision:** Add explicit `--reference-heading target`; keep `ego` as the backward-compatible default. Both methods use the same selected frame and spec. This remains a local straight-reference generator, not a route-following planner. Do not change target selection, add hidden rejection/resampling, relax constraints, or introduce Diffusion.

**Files:** `scene_parameters.py` owns the frame and consistent projected speed bounds; `run_scene_search.py` selects and records it in summary/JSON/NPZ; `scene_search.py` records actual acceleration extrema. Tests cover rotated actors, rigid rotation, legacy behavior, invalid modes, physical rejection, CLI and serialization.

**Environment:** This workspace has no `.git`; no worktree/commit can be created. Work stays in the supplied workspace, with original source snapshots and a patch for review. No subagents are needed for this tightly coupled correction.

- [x] Replay the original uploaded trials and verify input/map hashes.
- [x] Isolate the heading choice in a read-only experiment before modifying production code.
- [x] Add regression tests and observe expected failures.
- [x] Implement frame selection and diagnostic extrema, with the default mode unchanged.
- [x] Run all tests.
- [x] Validate both methods on all six real windows, 256 attempts each, seed 0.
- [x] Repeat a successful real run with the same seed; verify source hashes unchanged and serialized actors consistent.
- [x] Document exact counts, limitations, server commands, and package the minimal patch.

## Review focus

- Bounds and motion must use the same heading, including relative speed projection.
- The original ego states and masks must remain unchanged by frame selection.
- Default and explicit `ego` mode must reproduce the old physical trajectories.
- Valid candidates must still pass speed, acceleration, jerk, road and background-collision checks.
- A target heading is a local reference, not proof of route connectivity or actual simulation execution.

## Progress

- Original replay: six windows remain 0/256 for both methods. Failure sets match Linux in 3071/3072 trials; one differs near a numeric boundary. Linux uses Python 3.10.9/NumPy 1.23.5, local Python 3.8.3/NumPy 1.24.4. Candidate IDs are not assumed bit-identical across platforms.
- Target-heading diagnostic probe (Random only, not final acceptance): valid counts 22, 5, 11, 11, 2, 12 for windows 0,100,200,300,400,500. These are exploratory results; use the final verification report after implementation.
- TDD: the new regression file initially reported two assertion failures (CLI flag absent; acceleration extrema absent) and six missing-API errors. After the minimal implementation, all eight new tests passed. Full `python -m unittest discover -s tests`: 123 passed, 10.563 seconds.
- Formal real validation: Random valid counts 22/5/11/11/2/12; CEM 114/44/98/98/17/121. All six runs and w500 repeat exited 0. Exact within-environment trial trace repetition, JSON/NPZ roundtrips and unchanged source hash verified.
- Independent review: no blocking issue. Compared 200 default-mode candidates against snapshots with exact state/acceleration/jerk/ID equivalence. The direct nonworld-input provenance issue was reproduced by a failing test and fixed with an explicit world-frame guard. Final full suite: 124 passed, 8.830 seconds.
- Scope rulings: curved route planning, opposing traffic, continuous-time proofs and simulator execution remain outside this local-frame correction; no claim of those capabilities is made. The new world-frame guard only rejects inputs that could not pass the existing world-road evaluator.
- Final-source check: all 57 accepted exported candidates regenerated and passed the unchanged evaluator with matching states and risk. A static world-coordinate plot confirms the window500 Top-1 target/ego geometric contact, without claiming simulator execution.
- Packaging: patch dry-run and application against original snapshots both passed. All seven applied files match the final source after LF normalization. ZIP CRC and patch payload verified; source NPZ and final production source hashes rechecked.

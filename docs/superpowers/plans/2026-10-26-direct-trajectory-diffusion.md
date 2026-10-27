# Direct trajectory diffusion implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation with TDD and one independent final review.

**Goal:** Implement the user-approved fixed-index loader, minimal direct NPC diffusion training/sampling, real train/val evaluation entry points and candidate export.

**Architecture:** New standalone NumPy data/geometry modules and PyTorch denoiser/diffusion modules. Reuse the existing frozen manifests, normalizer and SUMO geometry conventions. Adapt the uploaded TemporalUnet structure while correcting the schedule/sample axes. No original Dreamer/RSSM or reference DiffScene source changes.

**Tech Stack:** Python >=3.8, NumPy, PyTorch; optional Matplotlib for plots. Test interpreter `D:/anaconda3/envs/omnisafe/python.exe` (Python3.8.20, torch2.3.0, CPU). Final composed test process uses numpy1.24.4 from base site-packages. No new dependencies installed.

**Spec:** `docs/diffscene_integration_review.md`; user explicitly accepted all three implementation steps after selecting direct trajectory diffusion.

## Global constraints

- Preserve canonical NPZs, fixed split memberships, existing normalizer and original code defaults.
- Conditions contain only historical ego/target/neighbors, masks and available static map/type metadata. Target future is a label, never a condition.
- H/F/dt come from the frozen dataset. Default existing data is 11/30/0.2; output `[B,K,F,2]` normalized ego-centric positions.
- Bounded per-worker episode cache; indexes store byte offsets instead of full per-window JSON/tensors.
- Training targets sampled reproducibly from eligible IDs; held-out target selection is deterministic from history. Use train only for optimization; val for model choice, test only by explicit request.
- Diffusion timesteps consistently 0..T-1. Implement noise prediction, DDPM and actual DDIM; no RSSM/LLM/risk guidance.
- Save config, complete normalizer parameters/ID, dataset hashes, seed and optimizer state with checkpoint. Paths are configurable/relocatable.
- Candidates preserve recorded history and position reference, mark generated source and simulation_verified=false. Report sampled kinematics/road checks separately from simulator truth.
- Full server pool is unavailable locally: local real single-episode checks are train-only smoke tests, not independent validation.
- No git metadata exists; deliver new files as a checked archive/patch, retaining original/reference hashes.

## Review focus

- Modified index/source/normalizer or cross-split IDs cannot be silently accepted.
- Future ego/neighbor/target values cannot enter the condition encoder or target-priority heuristic.
- Missing masks, all-missing neighbors/maps, yaw wrapping and zero-speed derivatives remain finite and retain missingness.
- Sampling shape, t=0/T-1 endpoints and DDIM skipping use matching schedules.
- Tiny train/val evaluation cannot be reported as full-pool generalization, and partial/failed training artifacts must not masquerade as successful checkpoints.

### Task 1: Data and coordinate adapter

Files: `embodied/envs/trajectory_dataset.py`, optional `trajectory_maps.py`, `tests/test_trajectory_dataset.py`.

Interfaces: `TrajectoryDataset(dataset_manifest, split, input_root=None, split_manifest=None, max_neighbors=8, cache_size=2, limit=None, seed=0, map_root=None)`; `set_epoch`, `__getitem__`, `__len__`, `xy_to_world`; `EpisodeSampler`; NumPy-only inspection CLI.

- [x] Write/run failing tests for fixed identity/hash checks, missing masks, target reproducibility, future-independent conditions, rotated coordinate round trips, relocation and bounded cache.
- [x] Implement index-byte offsets, per-episode resampling/cache, history conditions/label separation and optional static SUMO lane features.
- [x] Run data tests plus original suite; inspect actual uploaded NPZ using an explicitly train-only local manifest.

### Task 2: Direct denoiser and diffusion

Files: `trajectory_denoiser.py`, `trajectory_diffusion.py`, `tests/test_trajectory_diffusion.py`.

Interfaces: `ConditionalTrajectoryDenoiser(config, normalizer)` with `encode_condition`, `predict_noise`, `forward`; `TrajectoryDiffusion(...).training_loss(...)`, `.sample(condition,num_samples,method,steps,seed,eta)`.

- [x] Write/run RED tests for 30-frame shape, masked neighbors/maps, finite nonzero backward/update, analytical q/posterior endpoints, seeded DDPM/DDIM and batch/sample axes.
- [x] Adapt reference temporal residual/context structure; use explicit condition encoding and registered normalizer buffers.
- [x] Implement consistent diffusion math and unguided sampling; verify checkpoint/state round trip and no conditioning on labels.

### Task 3: Training, evaluation and candidates

Files: `train_trajectory_diffusion.py`, `sample_trajectories.py`, `trajectory_metrics.py`, `diffusion_candidate_adapter.py`, tests and `docs/long_tail_stage4.md`.

- [x] Write/run failing tests for runnable synthetic smoke, checkpoint validation, real-loader training/validation, metric definitions, history-only constant-velocity baseline, world/position-reference-preserving candidates and no simulator claims.
- [x] Add standalone CLI training with episode-local sampling, bounded limits, device selection, validation/checkpoint metadata and structured logs.
- [x] Add ADE/FDE (mean and best-of-K), diversity, elapsed sampling, provisional sampled motion checks and road checks only where geometry is available; save missingness/source definitions.
- [x] Add candidate export and optional standard plotting; no automatic simulator injection or replay writes.
- [x] Run 10–20 synthetic sample optimization/checkpoint/sample smoke and actual real single-episode train-only smoke; run the complete suite.

### Task 4: Final review and server delivery

- [x] Independent read-only whole-change review; regress important findings RED-to-GREEN.
- [x] Preserve original/reference hash evidence; document unavailable GPU/full server train/val results honestly.
- [x] Package new modules/tests/docs, apply/check archive against a clean snapshot, and provide exact server inspection/small-run/full-run commands.

## Execution ledger

Pre-flight: Task2 consumes only Task1's condition dictionaries/normalizer; Task3 explicitly separates labels/evaluation context from network conditions. Dataset identity and config travel in checkpoints. Task4 packages only the new modules/test/docs. No conflicting interfaces found. Existing approved review serves as design; no repeated approval is requested.

Task1 data regressions: 12 RED then GREEN, plus bounded-map-cache/fingerprint regression RED then GREEN; 13 focused tests pass. Real uploaded sample: inspected8 windows, labels[30,2], all conditions finite, world round-trip max error1.0767e-6m, one source load for adjacent windows. Map cache is bounded and current SUMO cfg/net bindings enter dataset identity.

Environment verification: basic NumPy environment lacks Torch; existing omnisafe environment provides Python3.8.20/Torch2.3.0/NumPy1.24.3 on CPU. Its first full-suite run failed 4 legacy imports because imageio was absent (test_danger_phase, test_danger_scenarios, test_sumo_episode_collection, test_world_model_adapter); no business code was changed to hide this. Re-running with the existing base site-packages in the test process PYTHONPATH, without installing/mutating either environment.

Task2 tensor regressions: 7 RED missing-module tests then GREEN with real CPU Torch. Checked30-frame forward, finite gradients and a real Adam update, exact posterior coefficient expectations, masked NaNs, zero neighbors, seeded DDPM/DDIM including one-step skip, and model-state sample round trip.

Task3 workflow regressions: 5 RED then GREEN, covering metric/oracle definitions, recorded-history candidate export, synthetic train/checkpoint/evaluation CLI, and frozen-loader train/val fixtures. The actual uploaded NPZ smoke is explicitly train-only: 16 selected windows, 40 optimization steps, no validation report. Synthetic training uses16 train samples and64 updates with4 held-out format samples. Both complete saved-model round trips and independent sampling/export.

Final independent review (review_direct_diffusion) requested two Important corrections: guard index contents throughout dataset lifetime against cross-split injection, and guard frozen map bindings before first load and on cached reuse. New data regressions failed before the fixes (index test plus two map subcases) and all15 focused data tests then passed. Review also marked candidate batch RNG context Minor; because reproducibility is an explicit contract, it was treated as Important for completion. The existing workflow test failed on missing sampling_context, then passed after exporting actual batch size/row/K/ordered dataset indices. No new scope was introduced and no repeat review is required by the execution workflow.

Fresh full verification after those fixes:182 tests passed in43.277s under the actual Torch environment with base site-packages added only to the test process PYTHONPATH. No environments were modified. Final real/synthetic workflow rerun passed all five commands; coordinate round trip max1.0813458857e-6m. The evidence collector initially used a wrong report key (generation instead of generation_metadata); corrected the collector and reran under a fresh output directory. Final evidence is docs/stage4_diffusion_validation/final_verified/verification.json, not the incomplete final_run directory.

Quality limits: synthetic val meanADE approximately4158m and real TRAIN-only subset meanADE approximately5610m, both motion_valid_rate0 after these short updates. Loss reduction and passing software tests do not constitute generation quality. No clipping/projection hides the poor samples. No full951-episode train/val/test run, GPU run or SUMO execution was performed. Server commands in docs/long_tail_stage4.md explicitly separate mechanics checks, subset monitoring and whole-split evaluation.

Delivery verified: artifacts/stage4_direct_diffusion.zip contains an additive13-file patch, README and final validation JSON, without checkpoints. GNU patch dry-run and real application to a temporary empty tree both returned0; every applied file matches the tested source (normalized LF). ZIP CRC passed. Existing original/reference/data hashes match both the initial smoke evidence and final run; all7 reference hashes also match the original static audit. Detailed check: docs/stage4_diffusion_validation/delivery_check.json. Server execution remains an explicitly documented external validation step because the full pool is not present locally.

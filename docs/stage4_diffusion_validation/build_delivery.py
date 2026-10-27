"""Build and check the additive Stage4 server patch, without trained weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARTIFACTS = ROOT/'artifacts'
PATCH_NAME = 'stage4_direct_diffusion.patch'
ZIP_NAME = 'stage4_direct_diffusion.zip'
FILES = [
    'embodied/envs/trajectory_maps.py',
    'embodied/envs/trajectory_dataset.py',
    'embodied/envs/trajectory_denoiser.py',
    'embodied/envs/trajectory_diffusion.py',
    'embodied/envs/trajectory_metrics.py',
    'embodied/envs/diffusion_candidate_adapter.py',
    'embodied/envs/trajectory_training.py',
    'embodied/envs/train_trajectory_diffusion.py',
    'embodied/envs/sample_trajectories.py',
    'tests/test_trajectory_dataset.py',
    'tests/test_trajectory_diffusion.py',
    'tests/test_trajectory_training.py',
    'docs/long_tail_stage4.md',
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    evidence_path = HERE/'final_verified/verification.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    original_evidence = json.loads((HERE/'verification.json').read_text(encoding='utf-8'))
    assert evidence['source_hashes_unchanged'] == original_evidence['source_hashes_unchanged']
    for path, expected in evidence['source_hashes_unchanged'].items():
        assert digest((ROOT/path).read_bytes()) == expected, path
    reference = json.loads((ROOT/'docs/diffscene_review/static_checks.json').read_text(encoding='utf-8'))
    for path, expected in reference['source_sha256'].items():
        assert digest((ROOT/'diffscene-main'/path).read_bytes()) == expected, path
    sources = {name: (ROOT/name).read_text(encoding='utf-8').replace('\r\n', '\n') for name in FILES}
    parts = []
    for name, source in sources.items():
        assert source.endswith('\n'), name
        lines = source.splitlines(keepends=True)
        parts.append('diff --git a/{0} b/{0}\nnew file mode 100644\n--- /dev/null\n+++ b/{0}\n@@ -0,0 +1,{1} @@\n'.format(name, len(lines)))
        parts.extend('+'+line for line in lines)
    patch = ''.join(parts).encode('utf-8')
    ARTIFACTS.mkdir(exist_ok=True)
    patch_path = ARTIFACTS/PATCH_NAME
    patch_path.write_bytes(patch)
    patch_executable = Path('C:/Program Files/Git/usr/bin/patch.exe')
    if not patch_executable.is_file():
        patch_executable = Path('patch')
    with tempfile.TemporaryDirectory(prefix='stage4_patch_check_') as directory:
        results = []
        for dry_run in (True, False):
            command = [str(patch_executable), '--batch', '--forward', '-p1', '-i', str(patch_path)]
            if dry_run:
                command.append('--dry-run')
            result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
            results.append({'dry_run': dry_run, 'returncode': result.returncode,
                            'stdout': result.stdout, 'stderr': result.stderr})
            assert result.returncode == 0, result.stdout+result.stderr
        for name, source in sources.items():
            assert (Path(directory)/name).read_text(encoding='utf-8') == source, name
    readme = '''Stage4 direct trajectory diffusion / 单 NPC 直接轨迹扩散

Prerequisite: the previous canonical episode split and window-preparation code,
the frozen split/window manifests, NumPy and PyTorch. Optional: Matplotlib.
No trained weights or original dataset files are included.

Run from the Linux project root (skip patch installation if full code was synced):
  unzip -o stage4_direct_diffusion.zip
  patch --dry-run --batch --forward -p1 < stage4_direct_diffusion.patch &&
  patch --batch --forward -p1 < stage4_direct_diffusion.patch
  python -m unittest discover -s tests

Then follow docs/long_tail_stage4.md, in this order:
  1. trajectory_dataset.py --inspect 16 on the existing frozen manifest;
  2. synthetic forward/backward/save/load/sampling smoke;
  3. a small real train/val run and sampling/export;
  4. full train and independent whole-val evaluation.

IMPORTANT: model quality has NOT passed. Local short-training samples have
large displacement errors and motion_valid_rate=0. No full951-episode real
train/val/test, GPU training or SUMO closed-loop execution was run locally.
Candidates are unverified trajectory references, not executable scenarios.
The source history, split, normalizer and original Dreamer code are preserved.

Validation:182 tests passed in43.277 seconds. Actual CPU Torch smoke includes
64 synthetic and40 real single-episode train-only optimization steps, exact
seeded checkpoint round trips, sampling and JSON/NPZ/PNG export. Detailed
evidence: stage4_validation.json (paths there describe the local evidence run).
Torch-dependent tests skip if Torch is absent; check that no such tests skip.
'''
    zip_path = ARTIFACTS/ZIP_NAME
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(PATCH_NAME, patch)
        archive.writestr('README_stage4.txt', readme.encode('utf-8'))
        archive.writestr('stage4_validation.json', evidence_path.read_bytes())
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert archive.read(PATCH_NAME) == patch
        assert archive.read('stage4_validation.json') == evidence_path.read_bytes()
    report = {'new_file_count': len(FILES), 'file_sha256_lf': {p: digest(s.encode('utf-8')) for p, s in sources.items()},
              'patch': str(patch_path), 'patch_sha256': digest(patch), 'patch_bytes': len(patch),
              'zip': str(zip_path), 'zip_sha256': digest(zip_path.read_bytes()), 'zip_bytes': zip_path.stat().st_size,
              'dry_run_and_apply': results, 'all_applied_files_match': True, 'archive_crc_passed': True,
              'original_and_reference_hashes_unchanged': True,
              'full_suite': {'tests': 182, 'seconds': 43.277, 'result': 'OK',
                             'command': 'python -m unittest discover -s tests'},
              'model_quality_passed': False, 'full_server_training_run': False}
    (HERE/'delivery_check.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('new_file_count', 'patch_bytes', 'zip_bytes',
                     'all_applied_files_match', 'archive_crc_passed', 'original_and_reference_hashes_unchanged')}, indent=2))


if __name__ == '__main__':
    main()

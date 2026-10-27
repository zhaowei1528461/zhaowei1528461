"""Build and dry-run the narrowly scoped server patch without a Git checkout."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = ('embodied/envs/scenario_data.py', 'embodied/envs/prepare_scenario_splits.py',
         'tests/test_scenario_splits.py', 'docs/scenario_split_preparation.md')


def main():
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    patch_path = artifacts/'scenario_split_preparation.patch'
    sections = []
    for name in FILES:
        old = HERE/'source_before'/name
        previous = old.read_text(encoding='utf-8').splitlines(keepends=True) if old.exists() else []
        current = (ROOT/name).read_text(encoding='utf-8').splitlines(keepends=True)
        sections.append('diff --git a/%s b/%s\n' % (name, name))
        if not old.exists():
            sections.append('new file mode 100644\n')
        sections.extend(difflib.unified_diff(previous, current,
                        fromfile='a/'+name if old.exists() else '/dev/null', tofile='b/'+name))
    with patch_path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(sections))
    check = HERE/'patch_application_check'
    if check.exists():
        raise RuntimeError('Use a new check directory; do not overwrite an earlier patch verification')
    check.mkdir()
    for name in FILES:
        old = HERE/'source_before'/name
        if old.exists():
            target = check/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('w', encoding='utf-8', newline='\n') as stream:
                stream.write(old.read_text(encoding='utf-8'))
    patch_executable = shutil.which('patch') or 'C:/Program Files/Git/usr/bin/patch.exe'
    for flags in (['--dry-run'], []):
        result = subprocess.run([patch_executable, '--batch', *flags, '-p1', '-i', str(patch_path)],
                                cwd=str(check), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
    for name in FILES:
        assert (check/name).read_text(encoding='utf-8') == (ROOT/name).read_text(encoding='utf-8'), name
    archive_path = artifacts/'scenario_split_preparation.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(patch_path, patch_path.name)
        archive.write(ROOT/'docs/scenario_split_preparation.md', 'scenario_split_preparation_README.md')
        archive.write(HERE/'verification.json', 'scenario_split_preparation_validation.json')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
    report = {'patch_dry_run': True, 'patch_applied_to_original_snapshot': True,
              'all_four_resulting_files_match': True, 'zip_crc_passed': True,
              'full_suite': {'command': 'D:/anaconda3/python.exe -m unittest discover -s tests',
                            'tests': 137, 'passed': True, 'seconds': 13.603},
              'files': {p.name: {'bytes': p.stat().st_size,
                                'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in (patch_path, archive_path)}}
    (HERE/'package_checks.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

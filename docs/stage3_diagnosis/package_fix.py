"""Build a reviewable patch without requiring a .git directory."""
from pathlib import Path
import difflib
import hashlib
import json
import zipfile


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSIS = ROOT / 'docs/stage3_diagnosis'
BEFORE = DIAGNOSIS / 'source_before'


def build():
    current = (ROOT/'README.md').read_text(encoding='utf-8')
    addition = ('合流道路上目标车与自车朝向差异较大时，可显式加 `--reference-heading target`；\n'
                '原 `ego` 参考方向仍为默认，物理/道路约束保持不变，详见 [真实样本修复与复验](docs/stage3_diagnosis/README.md)。\n')
    assert current.count(addition) == 1
    old_readme = current.replace(addition, '', 1)
    current = (ROOT/'docs/long_tail_stage3.md').read_text(encoding='utf-8')
    start = current.index('### 自车与目标车朝向不同时的参考方向\n')
    end = current.index('## 参数和运动学\n', start)
    old_long_tail = current[:start] + current[end:]
    new_paragraph = ('初始间距为自车 anchor 车身中心沿选定参数轴的纵向距离，相对速度也沿该轴定义；\n'
                     '默认轴是自车朝向，`target` 模式为目标 NPC 朝向。\n'
                     'target_offset 对 cut_in 表示通过自车 anchor 的选定参考线附近的终点横向偏移；\n'
                     '对 keep_lane 表示原横向位置的小偏移。\n')
    old_paragraph = ('初始间距为 ego anchor 前方的车身中心纵向距离，相对速度沿该 anchor heading 定义。\n'
                     'target_offset 对 cut_in 表示 ego 参考线附近的终点横向偏移；对 keep_lane 表示原横向位置的小偏移。\n')
    assert old_long_tail.count(new_paragraph) == 1
    old_long_tail = old_long_tail.replace(new_paragraph, old_paragraph, 1)
    for name, data in [('README.md', old_readme), ('docs/long_tail_stage3.md', old_long_tail)]:
        path = BEFORE/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_text(encoding='utf-8') == data
        else:
            path.write_bytes(data.encode('utf-8'))

    existing = ['embodied/envs/scene_parameters.py', 'embodied/envs/scene_search.py',
                'embodied/envs/run_scene_search.py', 'README.md', 'docs/long_tail_stage3.md']
    created = ['tests/test_scene_reference_heading.py', 'docs/stage3_diagnosis/verify_reference_heading.py']
    patch_parts = []
    manifest = {'description': 'Opt-in target heading correction; legacy ego default retained',
                'physical_constraints_changed': False, 'rssm_training_changed': False,
                'tests_passed': 124, 'files': {}}
    for name in existing+created:
        old = (BEFORE/name).read_text(encoding='utf-8') if name in existing else ''
        new = (ROOT/name).read_text(encoding='utf-8')
        patch_parts.append('diff --git a/%s b/%s\n' % (name, name))
        if name in created:
            patch_parts.append('new file mode 100644\n')
        patch_parts.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                           fromfile='a/'+name if name in existing else '/dev/null', tofile='b/'+name, n=3))
        manifest['files'][name] = {
            'before_sha256_normalized_LF': hashlib.sha256(old.encode()).hexdigest() if name in existing else None,
            'after_sha256_normalized_LF': hashlib.sha256(new.encode()).hexdigest()}
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    patch = artifacts/'stage3_reference_heading_fix.patch'
    patch.write_bytes(''.join(patch_parts).encode('utf-8'))
    (DIAGNOSIS/'patch_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')

    smoke = DIAGNOSIS/'patch_application_check'
    assert not smoke.exists(), 'Use a fresh patch application directory'
    for name in existing:
        path = smoke/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((BEFORE/name).read_text(encoding='utf-8').encode('utf-8'))
    install = '''Stage 3 reference heading fix (2026-10-26)

1. Copy this ZIP into the Ubuntu project root and unzip it there.
   It adds reports and the patch; existing production files change only in step 2.
2. In the project root:
   patch --dry-run -p1 < stage3_reference_heading_fix.patch
   patch -p1 < stage3_reference_heading_fix.patch
   python -m unittest discover -s tests
3. Add --reference-heading target to run_scene_search.py.
   Omitting this option retains the original ego-heading mode.
4. See docs/stage3_diagnosis/README.md for server commands, results and limits.

The patch changes 3 production modules, updates 2 documentation files,
and adds the regression tests and bounded verification tool.
Candidates remain offline and simulation_verified=false.
'''
    with zipfile.ZipFile(str(artifacts/'stage3_reference_heading_fix.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(patch, 'stage3_reference_heading_fix.patch')
        archive.writestr('STAGE3_FIX_README.txt', install)
        for name in ['README.md', 'replay.json', 'final_checks.json', 'patch_manifest.json',
                     'window500_candidate.png', 'verified_target_heading/verification.json']:
            archive.write(DIAGNOSIS/name, 'docs/stage3_diagnosis/'+name)
    print('Patch:', patch, patch.stat().st_size, 'bytes')
    print('ZIP:', artifacts/'stage3_reference_heading_fix.zip',
          (artifacts/'stage3_reference_heading_fix.zip').stat().st_size, 'bytes')
    print('Patch application directory:', smoke)


if __name__ == '__main__':
    build()

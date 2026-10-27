import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'embodied/envs/run_scene_search.py'
ROOT=SCRIPT.parents[2]


class SceneSearchCliTest(unittest.TestCase):
    def setUp(self): self.assertTrue(SCRIPT.is_file(),'stage 3 CLI missing')

    def run_cli(self,output,extra):
        return subprocess.run([sys.executable,str(SCRIPT),'--output',str(output),'--budget','12',
            '--cem-iterations','3','--top-k','2','--seed','9']+extra,cwd=str(ROOT),
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),capture_output=True,text=True)

    def test_demo_reports_equal_budget_and_roundtrip_scene_contract(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'demo'
            result=self.run_cli(output,['--demo'])
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['synthetic'])
            self.assertFalse(summary['simulation_verified'])
            for method in ('random','cem'):
                self.assertEqual(summary['searches'][method]['evaluations'],12)
                self.assertEqual(len((output/method/'trials.jsonl').read_text().splitlines()),12)
                scene=json.loads((output/method/'top_001.json').read_text(encoding='utf-8'))
                self.assertFalse(scene['simulation_verified'])
                self.assertEqual(scene['coordinate_frame'],'world')
                self.assertEqual(len(scene['target_trajectory']['states']),31)
                self.assertIn('source_map',scene)
                with np.load(output/method/'top_001.npz',allow_pickle=False) as archive:
                    self.assertEqual(archive['npc_ids'].tolist(),scene['npcs']['ids'])

    def test_real_npz_requires_explicit_map_without_silent_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            result=self.run_cli(Path(d)/'out',['--input','docs/6c873565d1ba7e753c679303.npz'])
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--sumo-cfg',result.stderr)

    def test_exported_candidate_ids_reproduce_in_separate_runs(self):
        with tempfile.TemporaryDirectory() as d:
            results=[]
            for name in ('a','b'):
                output=Path(d)/name
                result=self.run_cli(output,['--demo'])
                self.assertEqual(result.returncode,0,result.stderr+result.stdout)
                results.append(json.loads((output/'summary.json').read_text(encoding='utf-8')))
            for method in ('random','cem'):
                self.assertEqual(results[0]['searches'][method]['top_candidate_ids'],results[1]['searches'][method]['top_candidate_ids'])

    def test_all_invalid_is_nonzero_and_preserves_equal_budget_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            profile=json.loads((ROOT/'docs/stage3_search_config.json').read_text())
            profile['max_jerk']=1e-8
            (root/'profile.json').write_text(json.dumps(profile))
            result=self.run_cli(root/'out',['--demo','--profile',str(root/'profile.json')])
            self.assertEqual(result.returncode,2,result.stderr+result.stdout)
            summary=json.loads((root/'out/summary.json').read_text())
            for method in ('random','cem'):
                self.assertEqual(summary['searches'][method]['valid_count'],0)
                self.assertEqual(summary['searches'][method]['evaluations'],12)


if __name__=='__main__': unittest.main()

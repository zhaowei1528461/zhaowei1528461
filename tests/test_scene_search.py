from pathlib import Path
import sys
import unittest
import numpy as np

ENV_DIR=Path(__file__).resolve().parents[1]/'embodied/envs'
if str(ENV_DIR) not in sys.path: sys.path.insert(0,str(ENV_DIR))
from test_scene_generation import make_generator,parameters
try:
    import scene_search as search
except ImportError:
    search=None


class SceneSearchTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(search,'stage 3 search module missing')
        self.generator,self.road=make_generator()
        self.evaluator=search.SceneEvaluator(self.generator.spec.limits,self.road)

    def test_future_ego_contact_is_risk_but_not_invalid_physics(self):
        p=parameters(); p.update(initial_gap=6.,relative_speed=-2.)
        result=self.evaluator.evaluate(self.generator.generate(p,'cut_in'))
        self.assertTrue(result.valid,result.failures)
        self.assertGreater(result.risk_score,.5)
        self.assertTrue(result.risk_report['validity_flags']['geometric_overlap'])
        self.assertFalse(result.risk_report['validity_flags']['counterfactual_closed_loop'])

    def test_reverse_and_offroad_are_invalid(self):
        p=parameters(); p.update(relative_speed=-3.,longitudinal_accel=-2.)
        result=self.evaluator.evaluate(self.generator.generate(p,'cut_in'))
        self.assertFalse(result.valid)
        self.assertIn('reverse_motion',result.failures)
        self.evaluator.road=type(self.road).rectangle([-1.,-1.,1.,1.])
        result=self.evaluator.evaluate(self.generator.generate(parameters(),'cut_in'))
        self.assertFalse(result.valid)
        self.assertIn('outside_drivable_envelope',result.failures)

    def test_new_npc_background_collision_is_invalid(self):
        candidate=self.generator.generate(parameters(),'keep_lane')
        c=candidate.context
        c.npc_ids=('npc','background')
        c.history_npcs=np.repeat(c.history_npcs,2,axis=1)
        c.future_npcs=np.repeat(c.future_npcs,2,axis=1)
        c.future_npcs[:,1]=candidate.states[1:]
        c.npc_mask=np.ones((31,2),bool)
        candidate.trajectory.npc_ids=c.npc_ids
        candidate.trajectory.npc_states=np.repeat(candidate.trajectory.npc_states,2,axis=1)
        candidate.trajectory.npc_states[:,1]=candidate.states[1:]
        candidate.trajectory.npc_mask=c.npc_mask[1:]
        result=self.evaluator.evaluate(candidate)
        self.assertFalse(result.valid)
        self.assertIn('npc_background_overlap',result.failures)

    def test_both_searches_spend_exact_same_nondivisible_budget(self):
        for method in (search.RandomSearch(),search.CEMSearch(iterations=4)):
            result=method.run(self.generator,self.evaluator.evaluate,budget=17,seed=9,top_k=3)
            self.assertEqual(len(result.trials),17)
            self.assertEqual(result.summary['evaluations'],17)
            self.assertLessEqual(len(result.top_candidates),3)
            self.assertTrue(all(item[1].valid for item in result.top_candidates))

    def test_same_seed_reproduces_parameters_objectives_and_top_ids(self):
        for method in (search.RandomSearch(),search.CEMSearch(iterations=3)):
            first=method.run(self.generator,self.evaluator.evaluate,18,3,3)
            second=method.run(self.generator,self.evaluator.evaluate,18,3,3)
            self.assertEqual([r['parameters'] for r in first.trials],[r['parameters'] for r in second.trials])
            self.assertEqual([r['objective'] for r in first.trials],[r['objective'] for r in second.trials])
            self.assertEqual([c.candidate_id for c,r in first.top_candidates],[c.candidate_id for c,r in second.top_candidates])

    def test_local_rng_and_invalid_search_configuration(self):
        before=np.random.get_state()
        search.RandomSearch().run(self.generator,self.evaluator.evaluate,2,0,1)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        for kwargs in ({'iterations':0},{'elite_fraction':0.},{'min_std_fraction':0.}):
            with self.assertRaises(ValueError): search.CEMSearch(**kwargs)

    def test_all_invalid_has_no_top_candidates_and_keeps_budget(self):
        evaluator=search.SceneEvaluator(self.generator.spec.limits,type(self.road).rectangle([-1.,-1.,1.,1.]))
        result=search.CEMSearch(iterations=3).run(self.generator,evaluator.evaluate,11,0,3)
        self.assertEqual(len(result.trials),11)
        self.assertEqual(result.top_candidates,[])
        self.assertEqual(result.summary['valid_rate'],0.)

    def test_duplicate_trajectory_is_penalized_and_not_repeated_in_top_k(self):
        candidate=self.generator.generate(parameters(),'keep_lane')
        a=search.SearchArchive(top_k=3,duplicate_penalty=.1)
        first=self.evaluator.evaluate(candidate)
        first_objective=a.add(candidate,first)
        second=self.evaluator.evaluate(candidate)
        second_objective=a.add(candidate,second)
        self.assertAlmostEqual(first_objective-second_objective,.1)
        self.assertEqual(len(a.top()),1)

    def test_short_acceleration_ramp_cannot_hide_analytic_jerk_peak(self):
        generator=type(self.generator)(self.generator.window,0,self.generator.spec,
                                       acceleration_ramp=.01,validation_dt=.1)
        p=parameters(); p['longitudinal_accel']=.5
        result=self.evaluator.evaluate(generator.generate(p,'keep_lane'))
        self.assertFalse(result.valid)
        self.assertIn('jerk_limit',result.failures)
        self.assertAlmostEqual(result.metrics['max_abs_jerk'],93.75)

    def test_slow_lane_rejects_candidate_below_network_wide_limit(self):
        self.evaluator.road=type(self.road)(np.array([[-20.,3.5],[-20.,30.]]),
            np.array([[100.,3.5],[100.,30.]]),np.array([4.,4.]),[],
            'slow_and_fast_lanes',lane_speed=10.,segment_speeds=np.array([2.,10.]))
        result=self.evaluator.evaluate(self.generator.generate(parameters(),'keep_lane'))
        self.assertFalse(result.valid)
        self.assertIn('lane_speed_limit',result.failures)

    def test_exact_background_sample_is_not_corrupted_by_next_absent_frame(self):
        states=np.full((2,1,8),np.nan)
        states[0,0]=[0.,0.,1.,0.,0.,0.,4.,2.]
        sampled,valid=search._sample_observed(states,np.array([[True],[False]]),np.array([0.,.2]),np.array([0.]))
        self.assertTrue(valid[0,0])
        np.testing.assert_array_equal(sampled[0,0],states[0,0])

    def test_evaluator_errors_are_not_silently_counted_as_generation_failures(self):
        def broken_evaluator(candidate):
            raise ValueError('unexpected evaluation shape')
        with self.assertRaisesRegex(ValueError,'unexpected evaluation shape'):
            search.RandomSearch().run(self.generator,broken_evaluator,2,0,1)

    def test_generation_failures_respect_configured_invalid_penalty(self):
        self.generator.spec.bounds['lane_change_duration']=(10.,11.)
        result=search.RandomSearch(invalid_penalty=2.).run(self.generator,self.evaluator.evaluate,2,0,1)
        self.assertEqual([row['objective'] for row in result.trials],[-2.,-2.])


if __name__=='__main__': unittest.main()

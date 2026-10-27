import copy
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np

ENV_DIR = Path(__file__).resolve().parents[1]/'embodied/envs'
if str(ENV_DIR) not in sys.path: sys.path.insert(0, str(ENV_DIR))
from test_scenario_risk import vehicle, window
try:
    import scene_parameters as scenes
    import scene_road as roads
except ImportError:
    scenes = roads = None


def source_window():
    w = window(vehicle(vx=5.), vehicle(x=15., y=3.5, vx=5.))
    w.source_type = 'synthetic'
    w.source_path = None
    return w


def make_generator():
    w = source_window()
    limits = scenes.DynamicsLimits(max_speed=10., max_accel=2.6, max_decel=4.5)
    spec = scenes.SceneParameterSpec.from_window(w, 0, limits)
    return scenes.SceneGenerator(w, 0, spec), roads.RoadEnvelope.rectangle([-20., -2., 100., 5.5])


def parameters():
    return dict(initial_gap=10., relative_speed=-1., trigger_time=.5,
                longitudinal_accel=0., target_offset=0., lane_change_duration=4.)


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(scenes, 'stage 3 generation module missing')

    def test_quintic_meets_general_position_velocity_acceleration_boundaries(self):
        coeff = scenes.quintic_coefficients(2., .4, .2, -1., 0., 0., 3.)
        p,v,a,j = scenes.polynomial_motion(coeff, np.array([0.,3.]))
        np.testing.assert_allclose(p, [2.,-1.], atol=1e-10)
        np.testing.assert_allclose(v, [.4,0.], atol=1e-10)
        np.testing.assert_allclose(a, [.2,0.], atol=1e-10)

    def test_initial_gap_speed_and_lateral_end_are_physical(self):
        generator, _ = make_generator()
        c = generator.generate(parameters(), 'cut_in')
        self.assertAlmostEqual(c.states[0,0], 10.)
        self.assertAlmostEqual(c.states[0,2], 4.)
        self.assertAlmostEqual(c.states[0,1], 3.5)
        self.assertAlmostEqual(c.states[-1,1], 0.)
        self.assertAlmostEqual(c.states[-1,3], 0.)
        self.assertFalse(c.simulation_verified)

    def test_trigger_and_end_acceleration_are_continuous(self):
        generator, _ = make_generator()
        p=parameters(); p['longitudinal_accel']=.5
        t=np.array([.5-1e-7,.5,.5+1e-7,4.5-1e-7,4.5,4.5+1e-7])
        state, accel, jerk = generator.motion(p, 'cut_in', t)
        self.assertLess(abs(accel[2,1]-accel[0,1]), 1e-5)
        self.assertLess(abs(accel[-1,1]-accel[-3,1]), 1e-5)
        self.assertLess(abs(accel[2,0]-accel[0,0]), 1e-5)

    def test_candidate_does_not_modify_source_and_uses_synthetic_history(self):
        generator, _ = make_generator()
        original=generator.window.history_npcs.copy()
        c=generator.generate(parameters(),'cut_in')
        np.testing.assert_array_equal(original,generator.window.history_npcs)
        self.assertAlmostEqual(c.context.history_npcs[-1,0,0],10.)
        self.assertAlmostEqual(c.trajectory.npc_states[0,0,0],10.8)
        self.assertTrue(c.context.metadata['generated_target_history'])

    def test_invalid_parameters_and_nonfinite_inputs_are_rejected(self):
        generator, _ = make_generator()
        for p in ({**parameters(),'initial_gap':float('nan')},
                  {**parameters(),'lane_change_duration':-1.},
                  {**parameters(),'trigger_time':5.,'lane_change_duration':4.}):
            with self.assertRaises(ValueError): generator.generate(p,'cut_in')
        with self.assertRaises(ValueError): generator.generate(parameters(),'teleport')

    def test_front_bumper_source_is_converted_before_generation(self):
        w=source_window(); w.metadata['position_reference']='front_bumper_center'
        spec=scenes.SceneParameterSpec.from_window(w,0,scenes.DynamicsLimits(10.,2.6,4.5))
        generator=scenes.SceneGenerator(w,0,spec)
        c=generator.generate(parameters(),'cut_in')
        self.assertEqual(c.context.metadata['position_reference'],'center')
        self.assertAlmostEqual(c.states[0,0]-c.context.history_ego[-1,0],10.)

    def test_rectangle_checks_full_footprint_not_just_center(self):
        road=roads.RoadEnvelope.rectangle([-10.,-2.,100.,5.5])
        self.assertTrue(road.footprint_valid(np.array([vehicle(y=0.)]), .5))
        self.assertFalse(road.footprint_valid(np.array([vehicle(y=-1.5)]), .5))

    def test_sumo_config_resolves_relative_map_and_type_limits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'map.net.xml').write_text('<net><edge id="e"><lane id="e_0" speed="9" width="4" shape="-20,0 100,0"/></edge></net>')
            (root/'cars.rou.xml').write_text('<routes><vType id="car" maxSpeed="7" accel="2" decel="4"/><vehicle id="n" type="car"/></routes>')
            (root/'scene.sumocfg').write_text('<configuration><input><net-file value="map.net.xml"/><route-files value="cars.rou.xml"/></input></configuration>')
            road,limits,metadata=roads.load_sumo_scene(root/'scene.sumocfg','n',{})
            self.assertEqual(limits.max_speed,7.)
            self.assertEqual(limits.max_accel,2.)
            self.assertTrue(road.contains(np.array([[0.,0.]]))[0])
            self.assertFalse(road.contains(np.array([[0.,5.]]))[0])
            self.assertIn('net_sha256',metadata)

    def test_speed_limit_comes_from_the_matched_lane_not_network_maximum(self):
        road=roads.RoadEnvelope(np.array([[-20.,0.],[-20.,4.]]),
            np.array([[100.,0.],[100.,4.]]),np.array([4.,4.]),[], 'two_lanes',
            lane_speed=12.,segment_speeds=np.array([3.,12.]))
        np.testing.assert_array_equal(road.speed_limits(np.array([[0.,0.],[0.,4.]])),[3.,12.])

    def test_different_ramp_trajectories_have_different_candidate_ids(self):
        generator,_=make_generator(); p=parameters(); p['longitudinal_accel']=.5
        other=scenes.SceneGenerator(generator.window,0,generator.spec,acceleration_ramp=.2)
        first=generator.generate(p,'keep_lane'); second=other.generate(p,'keep_lane')
        self.assertNotEqual(first.states[-1,0],second.states[-1,0])
        self.assertNotEqual(first.candidate_id,second.candidate_id)

    def test_nearer_centerline_outside_its_lane_cannot_override_slow_lane(self):
        road=roads.RoadEnvelope(np.array([[-20.,1.2],[-20.,5.2]]),
            np.array([[100.,1.2],[100.,5.2]]),np.array([5.,3.]),[],
            'variable_width_lanes',lane_speed=10.,segment_speeds=np.array([2.,10.]))
        # y=3.5 belongs to the slow wide lane; the fast lane starts at y=3.7.
        self.assertEqual(road.speed_limits(np.array([[0.,3.5]]))[0],2.)


if __name__=='__main__': unittest.main()

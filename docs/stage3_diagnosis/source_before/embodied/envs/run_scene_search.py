"""Stage 3 offline scene search; candidates still require simulator execution."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import numpy as np
try:
    from .scenario_data import episode_from_record,load_episode,make_windows,_json_safe
    from .scene_parameters import SceneParameterSpec,SceneGenerator,DynamicsLimits,center_states
    from .scene_road import RoadEnvelope,load_sumo_scene
    from .scene_search import SceneEvaluator,RandomSearch,CEMSearch
    from .rssm_inference import file_sha256
    from .world_model_adapter import WorldModelAdapter
except ImportError:
    from scenario_data import episode_from_record,load_episode,make_windows,_json_safe
    from scene_parameters import SceneParameterSpec,SceneGenerator,DynamicsLimits,center_states
    from scene_road import RoadEnvelope,load_sumo_scene
    from scene_search import SceneEvaluator,RandomSearch,CEMSearch
    from rssm_inference import file_sha256
    from world_model_adapter import WorldModelAdapter


def _json(value):
    return json.dumps(_json_safe(value),ensure_ascii=False,sort_keys=True,allow_nan=False)


def demo_episode(history=11,future=30,dt=.2):
    frames=[]
    for i in range(history+future):
        t=i*dt
        frames.append({'sim_time':t,'cost':0.,'vehicles':[
            {'id':'Auto','pos':[5*t,0.],'v':5.,'yaw':0.,'a':0.,'length':5.,'width':2.},
            {'id':'target','pos':[15+5*t,3.5],'v':5.,'yaw':0.,'a':0.,'length':4.76,'width':1.95}]})
    return episode_from_record({'episode_id':'stage3_synthetic','scenario_id':'explicit_two_lane_demo',
        'source_type':'synthetic','metadata':{'position_reference':'center','prediction_source':'synthetic_kinematics'},
        'replay_trajectory':{'frames':frames}})


def choose_target(window,target_id=None):
    h=len(window.history_ego)
    eligible=[j for j in range(len(window.npc_ids)) if window.npc_mask[h-1:,j].all()]
    if target_id is not None:
        if target_id not in window.npc_ids: raise ValueError('Unknown --target-id')
        j=window.npc_ids.index(target_id)
        if j not in eligible: raise ValueError('Target needs all future observations; choose another window/target')
        return j
    if not eligible: raise ValueError('No target with complete anchor/future track')
    reference=window.metadata.get('position_reference')
    ego=center_states(window.history_ego[-1],reference)
    lateral=np.array([-np.sin(ego[4]),np.cos(ego[4])])
    def priority(j):
        npc=center_states(window.history_npcs[-1,j],reference)
        offset=abs(float((npc[:2]-ego[:2])@lateral))
        return (not .5<=offset<=12.,float(np.linalg.norm(npc[:2]-ego[:2])),window.npc_ids[j])
    return min(eligible,key=priority)


def export_candidate(candidate,evaluation,seed,map_source,source):
    c=candidate; context=c.context
    ego=np.concatenate([context.history_ego[-1:],context.future_ego])
    npcs=np.concatenate([context.history_npcs[-1:],c.trajectory.npc_states])
    masks=np.concatenate([context.npc_mask[len(context.history_ego)-1:len(context.history_ego)],c.trajectory.npc_mask])
    return {'schema_version':1,'format':'long_tail_candidate_scene','candidate_id':c.candidate_id,
        'parent_scenario_id':context.scenario_id,'source_episode_id':context.episode_id,
        'source':source,'source_map':map_source,'seed':seed,'behavior':c.behavior,'parameters':c.parameters,
        'coordinate_frame':'world','position_reference':'center',
        'units':{'position':'m','time':'s','velocity':'m/s','yaw':'rad'},
        'state_fields':['x','y','vx','vy','yaw','a','length','width'],
        'relative_timestamps':c.times,'anchor_timestamp':context.anchor_timestamp,
        'target_id':c.target_id,'initial_state':c.states[0],
        'target_trajectory':{'states':c.states,'mask':np.ones(len(c.states),bool)},
        'ego':{'id':'Auto','states':ego,'controller':'fixed_observed_reference'},
        'npcs':{'ids':context.npc_ids,'states':npcs,'mask':masks},
        'risk':evaluation.risk_report,'constraints':evaluation.metrics,
        'offline_constraints_passed':evaluation.valid,'simulation_verified':False,
        'rssm_counterfactual_supported':False,'generated_target_history':True,
        'generation_settings':context.metadata['generation_settings'],
        'execution_requirements':['Supply simulator map binding and initialization at anchor.',
                                  'Provide a closed-loop ego controller and validate NPC execution.',
                                  'Convert SUMO coordinates to CARLA coordinates if using CARLA.'],
        'native_simulator_configuration':False}


def run(args):
    output=Path(args.output)
    if output.exists() and any(output.iterdir()): raise ValueError('Output directory must be empty or new; use a new run directory')
    profile_path=Path(args.profile) if args.profile else Path(__file__).resolve().parents[2]/'docs/stage3_search_config.json'
    profile=json.loads(profile_path.read_text(encoding='utf-8'))
    allowed={'max_lateral_accel','max_jerk','max_yaw_rate','acceleration_ramp','validation_dt',
             'road_grid_spacing','default_lane_width','dedup_resolution','invalid_penalty','duplicate_penalty'}
    if set(profile)-allowed: raise ValueError('Unknown profile keys: '+str(set(profile)-allowed))
    if any(not np.isfinite(float(v)) or float(v)<0 or (float(v)==0 and k!='duplicate_penalty') for k,v in profile.items()):
        raise ValueError('Profile values must be finite and positive (duplicate penalty may be zero)')
    if not args.demo and not args.sumo_cfg: raise ValueError('Real --input requires explicit matching --sumo-cfg')
    if args.demo:
        episode=demo_episode(args.history,args.future)
        road=RoadEnvelope.rectangle([-30.,-2.,150.,5.5])
        limits=DynamicsLimits(10.,2.6,4.5,**{k:profile[k] for k in ('max_lateral_accel','max_jerk','max_yaw_rate')})
        map_source={'type':'explicit_synthetic_rectangle','bounds':[-30.,-2.,150.,5.5]}
        source={'synthetic':True,'input_path':None,'input_sha256':None}
    else:
        episode=load_episode(args.input)
        source={'synthetic':False,'input_path':str(args.input),'input_sha256':file_sha256(args.input),
                'recorded_source_cfg':episode.source_cfg,'simulation_seed':episode.seed}
    if episode.dt is None: raise ValueError('Uniform native episode timestep required')
    windows=make_windows(episode,history=args.history,future=args.future,dt=episode.dt,coordinate_frame='world')
    if not 0<=args.window_index<len(windows): raise ValueError('--window-index outside available complete windows')
    window=windows[args.window_index]; index=choose_target(window,args.target_id)
    if not args.demo: road,limits,map_source=load_sumo_scene(args.sumo_cfg,window.npc_ids[index],profile)
    spec=SceneParameterSpec.from_window(window,index,limits)
    generator=SceneGenerator(window,index,spec,profile.get('acceleration_ramp',1.),profile.get('validation_dt',.1))
    radius=spec.bounds['initial_gap'][1]+limits.max_speed*generator.horizon+10.
    road=road.local(generator.origin,radius)
    evaluator=SceneEvaluator(limits,road,road_grid_spacing=profile.get('road_grid_spacing',.5),
        dedup_resolution=profile.get('dedup_resolution',.5),invalid_penalty=profile.get('invalid_penalty',1.))
    methods=[RandomSearch(profile.get('duplicate_penalty',.1),profile.get('invalid_penalty',1.)),
             CEMSearch(args.cem_iterations,args.elite_fraction,args.smoothing,args.min_std_fraction,
                       duplicate_penalty=profile.get('duplicate_penalty',.1),invalid_penalty=profile.get('invalid_penalty',1.))]
    output.mkdir(parents=True,exist_ok=True)
    summaries={}
    for method in methods:
        result=method.run(generator,evaluator.evaluate,args.budget,args.seed,args.top_k)
        summaries[method.name]=result.summary
        folder=output/method.name; folder.mkdir(exist_ok=True)
        with (folder/'trials.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
            for trial in result.trials: stream.write(_json(trial)+'\n')
        for rank,(candidate,evaluation) in enumerate(result.top_candidates,1):
            payload=export_candidate(candidate,evaluation,args.seed,map_source,source)
            payload['search_provenance']={'method':method.name,'profile':profile,
                'profile_sha256':file_sha256(profile_path),'parameter_bounds':spec.bounds,
                'dynamics_limits':asdict(limits),'risk_config':asdict(evaluator.rules.config)}
            name='top_%03d'%rank
            (folder/(name+'.json')).write_text(_json(payload)+'\n',encoding='utf-8')
            np.savez_compressed(str(folder/(name+'.npz')),timestamps=candidate.times,states=candidate.states,
                npc_states=payload['npcs']['states'],npc_valid_mask=payload['npcs']['mask'],
                npc_ids=np.asarray(payload['npcs']['ids']),ego_id=np.asarray(payload['ego']['id']),
                ego_state=payload['ego']['states'],metadata_json=np.asarray(_json({k:v for k,v in payload.items()
                    if k not in ('npcs','ego','target_trajectory')})))
        print('%s: %d/%d offline valid, %d unique Top-K; %.3fs' %
            (method.name,result.summary['valid_count'],args.budget,len(result.top_candidates),result.summary['elapsed_seconds']))
    model_capability=WorldModelAdapter(episode).predict_future(window,npc_plan=np.empty((0,)))
    summary={'stage':3,'schema_version':1,'synthetic':bool(args.demo),'source':source,'source_map':map_source,
        'window_index':args.window_index,'anchor_timestamp':window.anchor_timestamp,'target_id':window.npc_ids[index],
        'dt':window.dt,'horizon':generator.horizon,'history':args.history,'future':args.future,
        'seed':args.seed,'budget_per_method':args.budget,'top_k_requested':args.top_k,
        'parameter_bounds':spec.bounds,'behaviors':spec.behaviors,'dynamics_limits':asdict(limits),
        'profile':profile,'profile_sha256':file_sha256(profile_path),'risk_config':asdict(evaluator.rules.config),
        'search_configuration':{m.name:vars(m) for m in methods},'searches':summaries,
        'rssm_candidate_prediction':model_capability,'simulation_verified':False,
        'source_map_identity_verified_against_recorded_hash':False,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
        'limitations':['Offline fixed observed ego response; no counterfactual RSSM prediction.',
            'Road checks sample lane buffers/junction polygons; not exact roadmesh or route connectivity.',
            'Generated initial conditions and compatible history are not recorded execution.',
            'JSON requires a simulator executor and map conversion; no SUMO/CARLA execution verified.']}
    (output/'summary.json').write_text(_json(summary)+'\n',encoding='utf-8')
    print('Report: '+str(output/'summary.json'))
    return 0 if all(v['top_candidate_ids'] for v in summaries.values()) else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true')
    source.add_argument('--input')
    parser.add_argument('--sumo-cfg')
    parser.add_argument('--output',required=True)
    parser.add_argument('--profile')
    parser.add_argument('--history',type=int,default=11)
    parser.add_argument('--future',type=int,default=30)
    parser.add_argument('--window-index',type=int,default=0)
    parser.add_argument('--target-id')
    parser.add_argument('--budget',type=int,default=32)
    parser.add_argument('--top-k',type=int,default=5)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--cem-iterations',type=int,default=4)
    parser.add_argument('--elite-fraction',type=float,default=.25)
    parser.add_argument('--smoothing',type=float,default=.7)
    parser.add_argument('--min-std-fraction',type=float,default=.05)
    try: return run(parser.parse_args(argv))
    except Exception as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr)
        return 1


if __name__=='__main__': sys.exit(main())

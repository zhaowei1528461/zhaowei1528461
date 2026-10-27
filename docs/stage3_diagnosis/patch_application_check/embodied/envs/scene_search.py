"""Shared offline constraints and equal-budget Random/CEM search."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict,dataclass,field
import hashlib
import math
import time
import numpy as np
try:
    from .scene_parameters import PARAMETERS
    from .scenario_risk import RiskEvaluator,obb_distance
except ImportError:
    from scene_parameters import PARAMETERS
    from scenario_risk import RiskEvaluator,obb_distance


@dataclass
class EvaluationResult:
    valid: bool
    risk_score: object
    objective: float
    failures: list
    fingerprint: str
    risk_report: object = None
    metrics: dict = field(default_factory=dict)


def _sample_observed(states,masks,times,query):
    index=np.searchsorted(times,query,side='right')-1
    index=np.clip(index,0,len(times)-1)
    following=np.minimum(index+1,len(times)-1)
    delta=times[following]-times[index]
    weight=np.divide(query-times[index],delta,out=np.zeros(len(query)),where=delta>1e-9)
    values=states[index]+(states[following]-states[index])*weight[:,None,None]
    # Exact source timestamps only need their own mask; interpolation needs both endpoints.
    exact=np.isclose(query,times[index],atol=1e-8,rtol=0)
    valid=masks[index] & (masks[following] | exact[:,None])
    values[exact]=states[index[exact]]
    yaw_delta=np.arctan2(np.sin(states[following,:,4]-states[index,:,4]),np.cos(states[following,:,4]-states[index,:,4]))
    values[:,:,4]=np.where(exact[:,None],states[index,:,4],states[index,:,4]+yaw_delta*weight[:,None])
    return values,valid


class SceneEvaluator:
    def __init__(self,limits,road,risk_config=None,road_grid_spacing=.5,dedup_resolution=.5,invalid_penalty=1.):
        if any(not np.isfinite(v) or v<=0 for v in (road_grid_spacing,dedup_resolution,invalid_penalty)):
            raise ValueError('Positive evaluation spacing/resolution/penalty required')
        self.limits,self.road=limits,road
        self.rules=RiskEvaluator(risk_config)
        self.road_grid_spacing,self.dedup_resolution=road_grid_spacing,dedup_resolution
        self.invalid_penalty=float(invalid_penalty)

    def evaluate(self,candidate):
        c=candidate; limits=self.limits; states=c.dense_states
        velocity,accel=states[:,2:4],c.accelerations
        speed=np.linalg.norm(velocity,axis=-1)
        heading=c.context.history_ego[-1,4]
        forward=velocity@np.array([np.cos(heading),np.sin(heading)])
        along=np.sum(velocity*accel,axis=-1)/np.maximum(speed,1e-9)
        lateral=np.abs(velocity[:,0]*accel[:,1]-velocity[:,1]*accel[:,0])/np.maximum(speed,1e-9)
        yaw_rate=lateral/np.maximum(speed,1e-9)
        max_jerk=float(np.linalg.norm(c.jerks,axis=-1).max())
        failures=[]
        checks={'reverse_motion':np.any(forward < -1e-7),
                'speed_limit':np.any(speed>limits.max_speed+1e-7),
                'longitudinal_acceleration_limit':np.any(along>limits.max_accel+1e-7) or np.any(along < -limits.max_decel-1e-7),
                'lateral_acceleration_limit':np.any(lateral>limits.max_lateral_accel+1e-7),
                'jerk_limit':max_jerk>limits.max_jerk+1e-7,
                'yaw_rate_limit':np.any(yaw_rate>limits.max_yaw_rate+1e-7)}
        failures.extend(k for k,v in checks.items() if v)
        local=(c.states[:,:2]-c.context.history_ego[-1,:2])@np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
        quantized=np.rint(local/self.dedup_resolution).astype('<i8')
        fingerprint=hashlib.sha256(quantized.tobytes()).hexdigest()
        metrics={'max_speed':float(speed.max()),'max_abs_jerk':max_jerk,
                 'min_longitudinal_accel':float(along.min()),
                 'max_longitudinal_accel':float(along.max()),
                 'min_forward_speed':float(forward.min()),
                 'max_lateral_accel':float(lateral.max()),'max_yaw_rate':float(yaw_rate.max()),
                 'road_checked':False,'road_check':'sampled_footprint_lane_envelope',
                 'road_grid_spacing':self.road_grid_spacing,
                 'max_validation_dt':float(np.diff(c.dense_times).max()),
                 'simulation_verified':False,'rssm_npc_conditioning_supported':False}
        if c.context.coordinate_frame!='world': failures.append('road_world_frame_required')
        if not failures:
            metrics['road_checked']=True
            if not self.road.footprint_valid(states,self.road_grid_spacing): failures.append('outside_drivable_envelope')
            lane_limits=self.road.speed_limits(states[:,:2],states[:,4])
            metrics['minimum_matched_lane_speed_limit']=float(lane_limits.min())
            if np.any(speed>lane_limits+1e-7): failures.append('lane_speed_limit')
        initial_ego=c.context.history_ego[-1]
        if obb_distance(c.states[0],initial_ego,'center')<=1e-9: failures.append('initial_ego_overlap')
        if not failures:
            h=len(c.context.history_ego)
            npc=np.concatenate([c.context.history_npcs[-1:],c.trajectory.npc_states])
            masks=c.context.npc_mask[h-1:].copy()
            masks[1:]=c.trajectory.npc_mask
            sampled,valid=_sample_observed(npc,masks,c.times,c.dense_times)
            for j in range(npc.shape[1]):
                if j==c.target_index: continue
                for i in np.flatnonzero(valid[:,j]):
                    other=sampled[i,j]
                    radius=.5*(np.linalg.norm(states[i,6:8])+np.linalg.norm(other[6:8]))
                    if np.linalg.norm(states[i,:2]-other[:2])<=radius and obb_distance(states[i],other,'center')<=1e-9:
                        failures.append('npc_background_overlap'); break
                if failures: break
        risk=None
        if not failures:
            risk=self.rules.evaluate_risk(c.context,c.trajectory)
            if not risk.validity_flags['valid'] or risk.score is None: failures.append('risk_input_incomplete')
        valid=not failures
        return EvaluationResult(valid,float(risk.score) if valid else None,
                                float(risk.score) if valid else -self.invalid_penalty,
                                failures,fingerprint,risk.to_dict() if risk is not None else None,metrics)


class SearchArchive:
    def __init__(self,top_k=5,duplicate_penalty=.1):
        if int(top_k)!=top_k or top_k<=0 or not np.isfinite(duplicate_penalty) or duplicate_penalty<0:
            raise ValueError('Positive top_k and nonnegative duplicate penalty required')
        self.top_k,self.duplicate_penalty=int(top_k),float(duplicate_penalty)
        self.seen=set(); self.best={}; self.duplicate_count=0

    def add(self,candidate,result):
        duplicate=result.fingerprint in self.seen
        self.seen.add(result.fingerprint)
        self.duplicate_count+=int(duplicate)
        objective=result.objective-(self.duplicate_penalty if duplicate else 0.)
        result.metrics['duplicate']=duplicate
        if result.valid:
            old=self.best.get(result.fingerprint)
            if old is None or result.risk_score>old[1].risk_score:
                self.best[result.fingerprint]=(candidate,result)
        return float(objective)

    def top(self):
        return sorted(self.best.values(),key=lambda item:(-item[1].risk_score,item[0].candidate_id))[:self.top_k]


@dataclass
class SearchResult:
    trials: list
    top_candidates: list
    summary: dict


class RandomSearch:
    name='random'
    def __init__(self,duplicate_penalty=.1,invalid_penalty=1.):
        if not np.isfinite(invalid_penalty) or invalid_penalty<=0:
            raise ValueError('Positive invalid penalty required')
        self.duplicate_penalty=duplicate_penalty
        self.invalid_penalty=float(invalid_penalty)

    def run(self,generator,evaluate,budget=64,seed=0,top_k=5):
        return _run(self,generator,evaluate,budget,seed,top_k)


class CEMSearch(RandomSearch):
    name='cem'
    def __init__(self,iterations=4,elite_fraction=.25,smoothing=.7,min_std_fraction=.05,
                 categorical_floor=.05,duplicate_penalty=.1,invalid_penalty=1.):
        super().__init__(duplicate_penalty,invalid_penalty)
        if int(iterations)!=iterations or iterations<=0: raise ValueError('Positive integer iterations required')
        if not 0<elite_fraction<=1 or not 0<smoothing<=1 or not 0<min_std_fraction<=1 or not 0<categorical_floor<.5:
            raise ValueError('Invalid CEM probabilities or variance floor')
        self.iterations=int(iterations); self.elite_fraction=float(elite_fraction)
        self.smoothing=float(smoothing); self.min_std_fraction=float(min_std_fraction)
        self.categorical_floor=float(categorical_floor); self.duplicate_penalty=duplicate_penalty


def _run(method,generator,evaluate,budget,seed,top_k):
    if int(budget)!=budget or budget<=0: raise ValueError('Positive integer budget required')
    budget=int(budget); iterations=method.iterations if isinstance(method,CEMSearch) else 1
    if iterations>budget: raise ValueError('CEM iterations cannot exceed evaluation budget')
    rng=np.random.default_rng(seed)
    archive=SearchArchive(top_k,method.duplicate_penalty)
    lower,upper=generator.spec.arrays(); span=upper-lower
    labels=tuple(generator.spec.behaviors); probabilities=np.ones(len(labels))/len(labels)
    means=np.tile((lower+upper)/2,(len(labels),1)); stds=np.tile(span/2,(len(labels),1))
    trials=[]; started=time.perf_counter(); failures=Counter(); valid_count=0
    for iteration in range(iterations):
        count=budget//iterations+int(iteration<budget%iterations)
        population=[]
        for _ in range(count):
            tick=time.perf_counter()
            label=int(rng.choice(len(labels),p=probabilities))
            vector=rng.uniform(lower,upper) if iteration==0 else np.clip(rng.normal(means[label],stds[label]),lower,upper)
            parameters=dict(zip(PARAMETERS,map(float,vector)))
            candidate=None
            try:
                candidate=generator.generate(parameters,labels[label])
            except ValueError as error:
                result=EvaluationResult(False,None,-method.invalid_penalty,['generation_error:'+str(error)],
                                        'invalid:'+str(len(trials)))
            else:
                result=evaluate(candidate)
            objective=archive.add(candidate,result)
            valid_count+=int(result.valid); failures.update(result.failures)
            trials.append({'index':len(trials),'iteration':iteration,'behavior':labels[label],
                           'parameters':parameters,'candidate_id':candidate.candidate_id if candidate is not None else None,
                           **asdict(result),'objective':objective,'elapsed_seconds':time.perf_counter()-tick})
            if result.valid: population.append((objective,label,vector))
        if isinstance(method,CEMSearch) and population:
            elites=sorted(population,key=lambda row:-row[0])[:max(1,int(math.ceil(count*method.elite_fraction)))]
            counts=np.bincount([row[1] for row in elites],minlength=len(labels)).astype(float)
            learned=counts/max(counts.sum(),1.)
            learned=(1-method.categorical_floor*len(labels))*learned+method.categorical_floor
            probabilities=(1-method.smoothing)*probabilities+method.smoothing*learned
            for label in range(len(labels)):
                chosen=np.array([row[2] for row in elites if row[1]==label])
                if len(chosen):
                    means[label]=(1-method.smoothing)*means[label]+method.smoothing*chosen.mean(0)
                    stds[label]=np.maximum((1-method.smoothing)*stds[label]+method.smoothing*chosen.std(0),method.min_std_fraction*span)
    top=archive.top()
    return SearchResult(trials,top,{'method':method.name,'seed':int(seed),'evaluations':len(trials),
        'valid_count':valid_count,'valid_rate':valid_count/budget,'duplicate_count':archive.duplicate_count,
        'unique_valid_count':len(archive.best),'failure_counts':dict(failures),
        'top_risks':[r.risk_score for c,r in top],'top_candidate_ids':[c.candidate_id for c,r in top],
        'elapsed_seconds':time.perf_counter()-started,'iterations':iterations,
        'final_behavior_probabilities':dict(zip(labels,map(float,probabilities))),
        'simulation_verified':False,'risk_source':'fixed_ego_candidate_rule'})

"""Deterministic stage 3 center-position trajectories, independent of training."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np
try:
    from .scenario_risk import CandidateTrajectory
except ImportError:
    from scenario_risk import CandidateTrajectory

PARAMETERS = ('initial_gap','relative_speed','trigger_time','longitudinal_accel','target_offset','lane_change_duration')
BEHAVIORS = ('cut_in','keep_lane')
REFERENCE_HEADINGS = ('ego', 'target')


@dataclass
class DynamicsLimits:
    max_speed: float
    max_accel: float
    max_decel: float
    max_lateral_accel: float = 2.5
    max_jerk: float = 8.0
    max_yaw_rate: float = .8

    def __post_init__(self):
        for name,value in vars(self).items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError('Dynamics limits must be finite and positive: '+name)


def center_states(states, reference):
    result=np.array(states,dtype=float,copy=True)
    if reference == 'front_bumper_center':
        result[...,0]-=.5*result[...,6]*np.cos(result[...,4])
        result[...,1]-=.5*result[...,6]*np.sin(result[...,4])
    elif reference != 'center':
        raise ValueError('Known position_reference required')
    return result


def quintic_coefficients(p0,v0,a0,p1,v1,a1,duration):
    if not np.isfinite([p0,v0,a0,p1,v1,a1,duration]).all() or duration <= 0:
        raise ValueError('Finite quintic boundaries and positive duration required')
    t=float(duration)
    matrix=np.array([[t**3,t**4,t**5],[3*t*t,4*t**3,5*t**4],[6*t,12*t*t,20*t**3]])
    rhs=np.array([p1-p0-v0*t-.5*a0*t*t,v1-v0-a0*t,a1-a0])
    return np.r_[p0,v0,.5*a0,np.linalg.solve(matrix,rhs)]


def polynomial_motion(coefficients,t):
    coefficients=np.asarray(coefficients,float)
    result=[]
    for derivative in range(4):
        result.append(np.polynomial.polynomial.polyval(t,coefficients))
        coefficients=np.arange(1,len(coefficients))*coefficients[1:]
    return tuple(result)


@dataclass
class SceneParameterSpec:
    bounds: dict
    limits: DynamicsLimits
    behaviors: tuple = BEHAVIORS
    reference_heading: str = 'ego'

    def __post_init__(self):
        if self.reference_heading not in REFERENCE_HEADINGS:
            raise ValueError('reference_heading must be ego or target')
        if set(self.bounds) != set(PARAMETERS): raise ValueError('Exactly six scene parameter bounds required')
        for name,pair in self.bounds.items():
            if len(pair)!=2 or not np.isfinite(pair).all() or pair[1]<=pair[0]:
                raise ValueError('Invalid parameter bounds: '+name)
        if not self.behaviors or any(v not in BEHAVIORS for v in self.behaviors):
            raise ValueError('Unknown behavior')

    @classmethod
    def from_window(cls,window,target_index,limits,reference_heading='ego'):
        if reference_heading not in REFERENCE_HEADINGS:
            raise ValueError('reference_heading must be ego or target')
        ego=center_states(window.history_ego[-1],window.metadata.get('position_reference'))
        npc=center_states(window.history_npcs[-1,target_index],window.metadata.get('position_reference'))
        if window.normalizer_id is not None or not np.isfinite(np.r_[ego,npc]).all():
            raise ValueError('Finite physical anchor states required')
        horizon=len(window.future_ego)*window.dt
        if horizon < 3.5: raise ValueError('At least 3.5 seconds future required for this search baseline')
        yaw=npc[4] if reference_heading=='target' else ego[4]
        heading=np.array([np.cos(yaw),np.sin(yaw)])
        ego_speed=float(ego[2:4]@heading)
        gap_min=.5*(ego[6]+npc[6])+1.
        upper_speed=min(2.,limits.max_speed-ego_speed)
        lower_speed=max(.2-ego_speed,min(-3.,upper_speed-.5))
        return cls({'initial_gap':(gap_min,max(gap_min+1.,min(30.,limits.max_speed*horizon))),
                    'relative_speed':(lower_speed,upper_speed),
                    'trigger_time':(0.,min(1.5,horizon/4)),
                    'longitudinal_accel':(-min(2.,limits.max_decel),min(1.,limits.max_accel)),
                    'target_offset':(-.3,.3),
                    'lane_change_duration':(2.5,min(4.5,horizon-min(1.5,horizon/4)))},limits,
                    reference_heading=reference_heading)

    def vector(self,parameters):
        if set(parameters)!=set(PARAMETERS): raise ValueError('Wrong scene parameter names')
        values=np.array([parameters[name] for name in PARAMETERS],float)
        lower,upper=self.arrays()
        if not np.isfinite(values).all() or np.any(values<lower-1e-9) or np.any(values>upper+1e-9):
            raise ValueError('Parameters outside configured bounds')
        return values

    def arrays(self):
        pairs=np.array([self.bounds[name] for name in PARAMETERS],float)
        return pairs[:,0],pairs[:,1]


@dataclass
class CandidateScenario:
    candidate_id: str
    parameters: dict
    behavior: str
    target_id: str
    target_index: int
    context: object
    trajectory: CandidateTrajectory
    times: np.ndarray
    states: np.ndarray  # anchor + future, [F+1,8], center/world
    dense_times: np.ndarray
    dense_states: np.ndarray
    accelerations: np.ndarray
    jerks: np.ndarray
    simulation_verified: bool = False


class SceneGenerator:
    def __init__(self,window,target_index,spec,acceleration_ramp=1.,validation_dt=.1):
        if window.coordinate_frame != 'world':
            raise ValueError('Scene generation requires world coordinates')
        if not 0<=target_index<len(window.npc_ids): raise ValueError('Invalid target NPC index')
        h=len(window.history_ego)
        if not window.npc_mask[h-1:,target_index].all() or not window.ego_mask[h-1:].all():
            raise ValueError('Target and ego need full anchor/future observations')
        if acceleration_ramp<=0 or validation_dt<=0 or not np.isfinite([acceleration_ramp,validation_dt]).all():
            raise ValueError('Positive ramp and validation dt required')
        self.window,self.target_index,self.spec=window,target_index,spec
        self.ramp,self.validation_dt=float(acceleration_ramp),min(float(validation_dt),window.dt)
        reference=window.metadata.get('position_reference')
        self.ego=center_states(window.history_ego[-1],reference)
        self.npc=center_states(window.history_npcs[-1,target_index],reference)
        yaw=self.npc[4] if spec.reference_heading=='target' else self.ego[4]
        self.rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        self.origin=self.ego[:2].copy()
        self.parameter_frame={'origin':self.origin.tolist(),'yaw_rad':float(yaw),
                              'heading_source':spec.reference_heading,'coordinate_frame':'world'}
        self.initial_lateral=float((self.npc[:2]-self.origin)@self.rotation[:,1])
        self.lateral_speed=float(self.npc[2:4]@self.rotation[:,1])
        self.ego_speed=float(self.ego[2:4]@self.rotation[:,0])
        self.horizon=len(window.future_ego)*window.dt

    def _lateral_polynomial(self,parameters,behavior):
        trigger=parameters['trigger_time']
        target=parameters['target_offset']+(self.initial_lateral if behavior=='keep_lane' else 0.)
        return quintic_coefficients(self.initial_lateral+self.lateral_speed*trigger,
            self.lateral_speed,0.,target,0.,0.,parameters['lane_change_duration'])

    def motion(self,parameters,behavior,times):
        self.spec.vector(parameters)
        if behavior not in self.spec.behaviors: raise ValueError('Unsupported behavior')
        trigger,duration=parameters['trigger_time'],parameters['lane_change_duration']
        if trigger+duration>self.horizon+1e-8: raise ValueError('Maneuver ends beyond horizon')
        times=np.asarray(times,float)
        if not np.isfinite(times).all(): raise ValueError('Invalid trajectory times')
        tau=times-trigger
        u=np.clip(tau/self.ramp,0.,1.); after=np.maximum(tau/self.ramp-1.,0.)
        smooth=10*u**3-15*u**4+6*u**5
        integral=2.5*u**4-3*u**5+u**6+after
        integral2=.5*u**5-.5*u**6+u**7/7+.5*after+.5*after**2
        jerk_shape=(30*u*u-60*u**3+30*u**4)/self.ramp
        acc=parameters['longitudinal_accel']; v0=self.ego_speed+parameters['relative_speed']
        x=parameters['initial_gap']+v0*times+acc*self.ramp**2*integral2
        vx=v0+acc*self.ramp*integral
        ax=acc*smooth; jx=acc*jerk_shape
        coeff=self._lateral_polynomial(parameters,behavior)
        y,vy,ay,jy=polynomial_motion(coeff,np.clip(tau,0.,duration))
        before=tau<0; done=tau>duration
        y=np.where(before,self.initial_lateral+self.lateral_speed*times,y)
        vy=np.where(before,self.lateral_speed,np.where(done,0.,vy))
        ay=np.where(before|done,0.,ay); jy=np.where(before|done,0.,jy)
        positions=np.stack([x,y],-1)@self.rotation.T+self.origin
        velocities=np.stack([vx,vy],-1)@self.rotation.T
        accelerations=np.stack([ax,ay],-1)@self.rotation.T
        jerks=np.stack([jx,jy],-1)@self.rotation.T
        speed=np.linalg.norm(velocities,axis=-1)
        yaw=np.unwrap(np.arctan2(velocities[:,1],velocities[:,0]))
        longitudinal=np.sum(velocities*accelerations,axis=-1)/np.maximum(speed,1e-9)
        states=np.column_stack([positions,velocities,yaw,longitudinal,
                                 np.full(len(times),self.npc[6]),np.full(len(times),self.npc[7])])
        return states,accelerations,jerks

    def generate(self,parameters,behavior):
        relative=np.asarray(self.window.timestamps)-self.window.anchor_timestamp
        full,_,_=self.motion(parameters,behavior,relative)
        h=len(self.window.history_ego)
        trigger,duration=parameters['trigger_time'],parameters['lane_change_duration']
        critical=[trigger,trigger+self.ramp/2,trigger+self.ramp,trigger+duration]
        coefficients=self._lateral_polynomial(parameters,behavior)
        # Include analytic axis-wise extrema as well as the regular grid.
        for derivative in range(1,5):
            roots=np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients,derivative))
            critical.extend(trigger+float(root.real) for root in roots
                            if abs(root.imag)<1e-8 and 0.<root.real<duration)
        dense=np.unique(np.r_[np.linspace(0.,self.horizon,int(np.ceil(self.horizon/self.validation_dt))+1),
                              [t for t in critical if 0.<=t<=self.horizon]])
        states,accel,jerk=self.motion(parameters,behavior,dense)
        context=copy.deepcopy(self.window)
        reference=context.metadata.get('position_reference')
        for name in ('history_ego','future_ego','history_npcs','future_npcs'):
            setattr(context,name,center_states(getattr(context,name),reference))
        context.metadata.update(position_reference='center',generated_target_history=True,
                                original_position_reference=reference,ego_response='fixed_observed_trajectory',
                                parameter_frame=dict(self.parameter_frame),
                                generation_settings={'acceleration_ramp':self.ramp,'validation_dt':self.validation_dt})
        context.history_npcs[:,self.target_index]=full[:h]
        context.npc_mask[:h,self.target_index]=True
        context.source_type='generated_candidate'
        future=context.future_npcs.copy()
        future[:,self.target_index]=full[h:]
        trajectory=CandidateTrajectory(future,context.npc_mask[h:].copy(),context.npc_ids,context.dt,context.coordinate_frame)
        identity=json.dumps({'parameters':parameters,'behavior':behavior,'episode':context.episode_id,
                             'anchor':context.anchor_timestamp,'target':context.npc_ids[self.target_index]},sort_keys=True)
        digest=hashlib.sha256(identity.encode()+full[h-1:].astype('<f8').tobytes()).hexdigest()[:24]
        return CandidateScenario(digest,dict(parameters),behavior,
            context.npc_ids[self.target_index],self.target_index,context,trajectory,relative[h-1:].copy(),
            full[h-1:].copy(),dense,states,accel,jerk)

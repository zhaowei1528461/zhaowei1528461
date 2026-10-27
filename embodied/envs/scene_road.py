"""Explicit sampled drivable envelopes; SUMO XML parsing needs no TraCI."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
try:
    from .scene_parameters import DynamicsLimits
    from .rssm_inference import file_sha256
except ImportError:
    from scene_parameters import DynamicsLimits
    from rssm_inference import file_sha256


def _inside_polygon(points,polygon):
    inside=np.zeros(len(points),bool)
    x,y=points[:,0],points[:,1]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        if abs(b[1]-a[1])<1e-12: continue
        crossing=((a[1]>y)!=(b[1]>y)) & (x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])
        inside^=crossing
    return inside


@dataclass
class RoadEnvelope:
    starts: np.ndarray
    ends: np.ndarray
    widths: np.ndarray
    polygons: list
    source: str
    lane_speed: float = 10.
    segment_speeds: object = None

    @classmethod
    def rectangle(cls,bounds):
        x0,y0,x1,y1=map(float,bounds)
        if not np.isfinite(bounds).all() or x1<=x0 or y1<=y0: raise ValueError('Invalid road rectangle')
        return cls(np.empty((0,2)),np.empty((0,2)),np.empty(0),
                   [np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])],'explicit_synthetic_rectangle')

    def contains(self,points):
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
            raise ValueError('Finite road points[M,2] required')
        result=np.zeros(len(points),bool)
        for poly in self.polygons:
            relevant=(points[:,0]>=poly[:,0].min())&(points[:,0]<=poly[:,0].max())&(points[:,1]>=poly[:,1].min())&(points[:,1]<=poly[:,1].max())
            if relevant.any(): result[relevant]|=_inside_polygon(points[relevant],poly)
        if len(self.starts):
            edges=self.ends-self.starts
            lengths=np.sum(edges**2,axis=-1)
            for start in range(0,len(points),128):
                offsets=points[start:start+128,None,:]-self.starts[None]
                fraction=np.clip(np.sum(offsets*edges[None],axis=-1)/lengths,0.,1.)
                squared=np.sum((offsets-fraction[...,None]*edges[None])**2,axis=-1)
                result[start:start+128]|=np.any(squared<=(self.widths[None]/2)**2+1e-9,axis=1)
        return result

    def footprint_valid(self,states,spacing=.5):
        if not np.isfinite(spacing) or spacing<=0: raise ValueError('Positive footprint spacing required')
        states=np.asarray(states,float)
        if states.ndim!=2 or states.shape[1]!=8 or not np.isfinite(states).all() or np.any(states[:,6:8]<=0):
            raise ValueError('Finite center vehicle states required')
        points=[]
        for state in states:
            length,width=state[6:8]
            x=np.unique(np.r_[np.linspace(-length/2,length/2,int(np.ceil(length/spacing))+1),0.])
            y=np.unique(np.r_[np.linspace(-width/2,width/2,int(np.ceil(width/spacing))+1),0.])
            local=np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2)
            c,s=np.cos(state[4]),np.sin(state[4])
            points.append(local@np.array([[c,s],[-s,c]])+state[:2])
        return bool(self.contains(np.concatenate(points)).all())

    def speed_limits(self,points,headings=None):
        """Containing aligned lanes; nearest fallback only inside junction polygons."""
        points=np.asarray(points,float)
        if not len(self.starts): return np.full(len(points),self.lane_speed)
        speeds=np.asarray(self.segment_speeds if self.segment_speeds is not None else
                          np.full(len(self.starts),self.lane_speed),float)
        edges=self.ends-self.starts
        offsets=points[:,None,:]-self.starts[None]
        fraction=np.clip(np.sum(offsets*edges[None],axis=-1)/np.sum(edges**2,axis=-1),0.,1.)
        distance=np.sum((offsets-fraction[...,None]*edges[None])**2,axis=-1)
        aligned=np.ones(distance.shape,bool)
        if headings is not None:
            forward=np.column_stack([np.cos(headings),np.sin(headings)])
            aligned=forward@edges.T>0
        containing=(distance<=(self.widths[None]/2)**2+1e-9)&aligned
        limits=np.min(np.where(containing,speeds[None],np.inf),axis=1)
        junction=np.zeros(len(points),bool)
        for poly in self.polygons: junction|=_inside_polygon(points,poly)
        fallback=~containing.any(axis=1)&junction
        if fallback.any():
            eligible=np.where(aligned[fallback],distance[fallback],np.inf)
            nearest=np.min(eligible,axis=1)
            matching=np.isfinite(eligible)&(eligible<=nearest[:,None]+1e-7)
            limits[fallback]=np.min(np.where(matching,speeds[None],np.inf),axis=1)
        return np.where(np.isfinite(limits),limits,0.)

    def local(self,origin,radius):
        if not len(self.starts): return self
        lo,hi=np.minimum(self.starts,self.ends),np.maximum(self.starts,self.ends)
        keep=np.all(hi>=np.asarray(origin)-radius-self.widths[:,None],axis=1)&np.all(lo<=np.asarray(origin)+radius+self.widths[:,None],axis=1)
        polygons=[p for p in self.polygons if np.all(p.max(0)>=np.asarray(origin)-radius) and np.all(p.min(0)<=np.asarray(origin)+radius)]
        speeds=np.asarray(self.segment_speeds)[keep] if self.segment_speeds is not None else None
        return RoadEnvelope(self.starts[keep],self.ends[keep],self.widths[keep],polygons,self.source,self.lane_speed,speeds)


def _shape(text):
    return np.array([[float(v) for v in point.split(',')[:2]] for point in text.split()],float)


def load_sumo_scene(config_path,target_id,profile):
    config_path=Path(config_path)
    config=ET.parse(str(config_path)).getroot()
    net_node=config.find('./input/net-file'); route_node=config.find('./input/route-files')
    if net_node is None or route_node is None: raise ValueError('SUMO cfg must name net-file and route-files')
    net_path=config_path.parent/net_node.attrib['value']
    net=ET.parse(str(net_path)).getroot()
    starts,ends,widths,speeds,segment_speeds=[],[],[],[],[]
    for lane in net.iter('lane'):
        allow=lane.get('allow','').split(); deny=lane.get('disallow','').split()
        if 'passenger' in deny or 'all' in deny or (allow and 'passenger' not in allow and 'all' not in allow): continue
        points=_shape(lane.get('shape',''))
        width=float(lane.get('width',profile.get('default_lane_width',3.2)))
        if width<=0 or len(points)<2: continue
        for a,b in zip(points[:-1],points[1:]):
            if np.linalg.norm(a-b)>1e-9:
                starts.append(a); ends.append(b); widths.append(width)
                segment_speeds.append(float(lane.get('speed','0')))
        speeds.append(float(lane.get('speed','0')))
    polygons=[_shape(node.get('shape','')) for node in net.iter('junction') if len(node.get('shape','').split())>=3]
    if not starts: raise ValueError('No passenger drivable lane geometry')
    types,vehicles,route_meta={}, {}, []
    for name in route_node.attrib['value'].split(','):
        path=config_path.parent/name.strip(); routes=ET.parse(str(path)).getroot()
        types.update({node.attrib['id']:node.attrib for node in routes.iter('vType')})
        vehicles.update({node.attrib['id']:node.attrib for node in routes.iter('vehicle')})
        route_meta.append({'path':str(path),'sha256':file_sha256(path)})
    vehicle=vehicles.get(target_id)
    if vehicle is None or vehicle.get('type') not in types: raise ValueError('Explicit target vType missing in route files')
    typ=types[vehicle['type']]
    required=('maxSpeed','accel','decel')
    if any(k not in typ for k in required): raise ValueError('Target vType needs explicit maxSpeed/accel/decel')
    limits=DynamicsLimits(min(float(typ['maxSpeed']),max(speeds)),float(typ['accel']),float(typ['decel']),
                           **{k:float(profile.get(k,v)) for k,v in [('max_lateral_accel',2.5),('max_jerk',8.),('max_yaw_rate',.8)]})
    road=RoadEnvelope(np.array(starts),np.array(ends),np.array(widths),polygons,
                      'sumo_lane_buffers_and_junction_polygons',max(speeds),np.array(segment_speeds))
    return road,limits,{'sumo_cfg':str(config_path),'cfg_sha256':file_sha256(config_path),
                       'net_path':str(net_path),'net_sha256':file_sha256(net_path),'routes':route_meta,
                       'target_type':vehicle['type'],'target_vtype':dict(typ),'route_connectivity_verified':False}

"""Optional static SUMO geometry, independent of Torch and simulator execution."""
from pathlib import Path
from collections import OrderedDict
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np

try:
    from .prepare_trajectory_windows import _hash
    from .scene_road import RoadEnvelope
except ImportError:
    from prepare_trajectory_windows import _hash
    from scene_road import RoadEnvelope


class SumoMapStore:
    def __init__(self, root=None, max_segments=64, radius=80., cache_size=4):
        self.root = Path(root).resolve() if root is not None else None
        self.max_segments, self.radius = max_segments, radius
        if cache_size < 1:
            raise ValueError('Map cache size must be positive')
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()
        self._bindings = {}

    def cache_info(self):
        return {'maps': len(self._cache), 'capacity': self.cache_size}

    def binding_fingerprint(self, source_cfgs):
        bindings, net_hashes = [], {}
        for source in sorted(set(s for s in source_cfgs if s)):
            if source in self._bindings:
                self._guard_binding(source)
                frozen = self._bindings[source]
                bindings.append((source, frozen['cfg_sha256'], frozen['net_sha256']) if frozen else (source, None, None))
                continue
            cfg = self.resolve(source)
            if cfg is None:
                self._bindings[source] = None
                bindings.append((source, None, None))
                continue
            cfg_signature = self._signature(cfg)
            cfg_hash = _hash(cfg)
            node = ET.parse(str(cfg)).getroot().find('./input/net-file')
            if node is None:
                raise ValueError('SUMO configuration has no net-file')
            net = (cfg.parent/node.attrib['value']).resolve()
            if net not in net_hashes:
                net_signature = self._signature(net)
                net_hashes[net] = (_hash(net), net_signature)
            net_hash, net_signature = net_hashes[net]
            self._bindings[source] = {'cfg_path': cfg, 'net_path': net, 'cfg_sha256': cfg_hash,
                 'net_sha256': net_hash, 'cfg_signature': cfg_signature, 'net_signature': net_signature}
            self._guard_binding(source)
            bindings.append((source, cfg_hash, net_hash))
        return hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _guard_binding(self, source):
        if not source:
            return None
        if source not in self._bindings:
            self.binding_fingerprint([source])
        binding = self._bindings[source]
        current = self.resolve(source)
        if binding is None:
            if current is not None:
                raise ValueError('Previously missing map binding changed')
            return None
        if (current != binding['cfg_path'] or self._signature(binding['cfg_path']) != binding['cfg_signature']
                or self._signature(binding['net_path']) != binding['net_signature']):
            raise ValueError('SUMO map changed after its binding hash was frozen')
        return binding

    def resolve(self, source_cfg):
        if not source_cfg:
            return None
        original = Path(source_cfg)
        candidates = ([self.root/original.name] if self.root else []) + [original]
        return next((p.resolve() for p in candidates if p.is_file()), None)

    def load(self, source_cfg):
        binding = self._guard_binding(source_cfg)
        if binding is None:
            return None
        cfg = binding['cfg_path']
        if cfg in self._cache:
            result = self._cache.pop(cfg)
            self._cache[cfg] = result
            return result
        node = ET.parse(str(cfg)).getroot().find('./input/net-file')
        if node is None:
            raise ValueError('SUMO configuration has no net-file')
        net_path = (cfg.parent/node.attrib['value']).resolve()
        net = ET.parse(str(net_path)).getroot()
        starts, ends, widths, speeds = [], [], [], []
        for lane in net.iter('lane'):
            allow, deny = lane.get('allow', '').split(), lane.get('disallow', '').split()
            if 'passenger' in deny or 'all' in deny or (allow and 'passenger' not in allow and 'all' not in allow):
                continue
            points = np.array([[float(v) for v in p.split(',')[:2]] for p in lane.get('shape', '').split()])
            width, speed = float(lane.get('width', '3.2')), float(lane.get('speed', '0'))
            if width <= 0 or speed < 0 or not np.isfinite([width, speed]).all():
                raise ValueError('Invalid SUMO lane width/speed')
            for a, b in zip(points[:-1], points[1:]):
                if not np.isfinite([a, b]).all():
                    raise ValueError('Nonfinite SUMO lane geometry')
                if np.linalg.norm(a-b) > 1e-9:
                    starts.append(a); ends.append(b); widths.append(width); speeds.append(speed)
        if not starts:
            raise ValueError('SUMO map has no passenger lane segments')
        polygons = [np.array([[float(v) for v in p.split(',')[:2]] for p in n.get('shape', '').split()])
                    for n in net.iter('junction') if len(n.get('shape', '').split()) >= 3]
        road = RoadEnvelope(np.asarray(starts), np.asarray(ends), np.asarray(widths), polygons,
                            'sumo_lane_buffers_and_junction_polygons', max(speeds), np.asarray(speeds))
        self._guard_binding(source_cfg)
        provenance = {'cfg_path': str(cfg), 'cfg_sha256': binding['cfg_sha256'], 'net_path': str(net_path),
                      'net_sha256': binding['net_sha256'], 'identity_verified_against_recording': False,
                      'feature_units': 'endpoints/50m, width/5m, speed_limit/30m_per_s'}
        self._cache[cfg] = (road, provenance)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return road, provenance

    def features(self, source_cfg, anchor):
        values = np.zeros((self.max_segments, 6), dtype=np.float32)
        mask = np.zeros(self.max_segments, dtype=bool)
        loaded = self.load(source_cfg)
        if loaded is None:
            return values, mask, {'available': False, 'reason': 'missing_map_binding'}
        road, provenance = loaded
        centers = (road.starts+road.ends)*.5
        distance = np.linalg.norm(centers-anchor[:2], axis=-1)
        eligible = np.flatnonzero(distance <= self.radius + np.linalg.norm(road.ends-road.starts, axis=-1)*.5)
        selected = eligible[np.argsort(distance[eligible], kind='stable')[:self.max_segments]]
        c, s = np.cos(anchor[4]), np.sin(anchor[4])
        rotation = np.array([[c, -s], [s, c]])
        n = len(selected)
        values[:n, :2] = ((road.starts[selected]-anchor[:2])@rotation)/50.
        values[:n, 2:4] = ((road.ends[selected]-anchor[:2])@rotation)/50.
        values[:n, 4] = road.widths[selected]/5.
        values[:n, 5] = road.segment_speeds[selected]/30.
        mask[:n] = True
        return values, mask, dict(provenance, available=bool(n))

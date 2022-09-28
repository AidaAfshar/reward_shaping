from typing import Dict, Any

import numpy as np

from reward_shaping.core.configs import TLRewardConfig


def _get_highway_default_monitoring_variables():
    return ['time', 'max_steps',
            'collision',
            'distance_to_target', 'target_distance_tol']


def _get_highway_default_monitoring_types():
    return ['int', 'int',
            'float',
            'float', 'float']


def _get_highway_default_monitoring_procedure(state, done, info):
    time_norm = state['time'] / info['max_steps']
    max_steps_norm = info['max_steps'] / info['max_steps']
    distance_to_target_norm = state['distance_to_target'] / info['x_limit']
    target_distance_tol_norm = info['target_distance_tol'] / info['x_limit']

    # compute monitoring variables
    monitored_state = {
        'time': time_norm,
        'max_steps': max_steps_norm,
        'collision': 1.0 if state['collision'] > 0 else -1.0,
        'distance_to_target': distance_to_target_norm,
        'target_distance_tol': target_distance_tol_norm,

    }
    return monitored_state


class HighwaySTLReward(TLRewardConfig):
    _safe_distance = "always(collision<=0)"
    _reach_target = "eventually(distance_to_target <= target_distance_tol)"

    @property
    def spec(self) -> str:
        safety_requirements = f"({self._safe_distance})"
        target_requirement = self._reach_target
        spec = f"({safety_requirements}) and ({target_requirement})"
        return spec

    @property
    def requirements_dict(self):
        return {'safe_distance': self._safe_distance,
                'reach_target': self._reach_target}

    @property
    def monitoring_variables(self):
        return _get_highway_default_monitoring_variables()

    @property
    def monitoring_types(self):
        return _get_highway_default_monitoring_types()

    def get_monitored_state(self, state, done, info) -> Dict[str, Any]:
        return _get_highway_default_monitoring_procedure(state, done, info)

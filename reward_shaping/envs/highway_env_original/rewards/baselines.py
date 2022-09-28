import math
import numpy as np

from reward_shaping.core.reward import RewardFunction
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.highway_env_RSS import highway_utils

from typing import List
from reward_shaping.core.configs import EvalConfig
from reward_shaping.core.helper_fns import monitor_stl_episode


class HighwaySparseTargetReward(RewardFunction):
    """
        reward(s,a) := bonus, if target is reached
        reward(s,a) := penalty for crash
    """

    def reached_target(self, state, info):
        assert 'distance_to_target' in state and 'TARGET_DISTANCE' in info
        return 1 - clip_and_norm(state['distance_to_target'], 0, info['TARGET_DISTANCE'])

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'observation' in state and 'done' in info
        assert 'collision' in state
        if info['done']:
            if state['collision'] == 1:
                return -1.0
            elif self.reached_target(state, info):
                return 1.0
        return 0


class HighwayProgressTargetReward(RewardFunction):
    """
    reward(s, a, s') := target(s') - target(s)
    """

    def target_potential(self, state, info):
        assert 'distance_to_target' in state and 'TARGET_DISTANCE' in info
        return 1 - clip_and_norm(state['distance_to_target'], 0, info['TARGET_DISTANCE'])

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        assert 'TARGET_DISTANCE' in info
        progress = self.target_potential(next_state, info) - self.target_potential(state, info)
        return progress


class HighwayEvalConfig(EvalConfig):

    def __init__(self, **kwargs):
        super(HighwayEvalConfig, self).__init__(**kwargs)
        self._max_episode_len = None

    @property
    def monitoring_variables(self) -> List[str]:
        return ['time', 'max_steps',
                'ego_x', 'ego_y', 'ego_vx', 'ego_vy',
                'ego_lane_index', 'lanes_count',
                'max_y', 'target_lane_y', 'target_lane_tol', 'lane_dif',
                'collision',
                'road_progress', 'distance_to_target', 'target_x', 'target_distance_tol',
                'x_limit', 'y_limit', 'vx_limit', 'vy_limit',
                'speed_lower_bound', 'speed_upper_bound', 'speed_tol', 'speed_dif']

    @property
    def monitoring_types(self) -> List[str]:
        return ['int', 'int',
                'float', 'float', 'float', 'float',
                'float', 'float',
                'float', 'float', 'float', 'float',
                'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float',
                'float', 'float', 'float', 'float']

    def get_monitored_state(self, state, done, info):
        monitored_state = {
            'time': state['time'],
            'max_steps': info['max_steps'],
            'ego_x': state['ego_x'],
            'ego_y': state['ego_y'],
            'ego_vx': state['ego_vx'],
            'ego_vy': state['ego_vy'],
            'ego_lane_index': state['ego_lane_index'],
            'lanes_count': info['lanes_count'],
            'max_y': info['max_y'],
            'target_lane_y': info['target_lane_y'],
            'target_lane_tol': info['target_lane_tol'],
            'lane_dif': abs(state['ego_y'] - info['target_lane_y']),
            'collision': state['collision'],
            'road_progress': state['road_progress'],
            'distance_to_target': state['distance_to_target'],
            'target_x': info['target_x'],
            'target_distance_tol': info['target_distance_tol'],
            'x_limit': info['x_limit'],
            'y_limit': info['y_limit'],
            'vx_limit': info['vx_limit'],
            'vy_limit': info['vy_limit'],
            'speed_lower_bound': info['speed_lower_bound'],
            'speed_upper_bound': info['speed_upper_bound'],
            'speed_tol': info['speed_tol'],
            'speed_dif': abs(info['speed_upper_bound'] - state['ego_vx']),
        }
        self._max_episode_len = info['max_steps']
        return monitored_state

    def eval_episode(self, episode) -> float:
        # discard any eventual prefix
        i_init = np.nonzero(episode['time'] == np.min(episode['time']))[-1][-1]
        episode = {k: list(l)[i_init:] for k, l in episode.items()}
        #
        safety_spec = "always(collision <= 0)"
        safety_rho = monitor_stl_episode(stl_spec=safety_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        target_spec = "eventually(distance_to_target <= target_distance_tol)"
        target_rho = monitor_stl_episode(stl_spec=target_spec,
                                         vars=self.monitoring_variables, types=self.monitoring_types,
                                         episode=episode)[0][1]
        comfort_highspeed_spec = "(speed_dif <= speed_tol)"
        comfort_rightlane_spec = "(lane_dif <= target_lane_tol)"
        comfort_metrics = []
        for comfort_spec in [comfort_highspeed_spec, comfort_rightlane_spec]:
            comfort_trace = monitor_stl_episode(stl_spec=comfort_spec,
                                                vars=self.monitoring_variables, types=self.monitoring_types,
                                                episode=episode)
            comfort_trace = comfort_trace + [[-1, -1] for _ in
                                             range((self._max_episode_len - len(comfort_trace)))]
            comfort_mean = np.mean([float(rob >= 0) for t, rob in comfort_trace])
            comfort_metrics.append(comfort_mean)
        #
        tot_score = float(safety_rho >= 0) + 0.5 * float(target_rho >= 0) + 0.25 * comfort_mean
        return tot_score

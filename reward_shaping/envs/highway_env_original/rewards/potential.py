import math
from reward_shaping.core.reward import RewardFunction
from typing import List
from typing import Union
import numpy as np
from reward_shaping.core.utils import clip_and_norm
from reward_shaping.envs.highway_env_original.specs import get_all_specs

gamma = 1.0


def safety_no_collision_potential(state, info):
    assert 'collision' in state
    return 0 if (state['collision'] == 1) else 1


def target_reach_potential(state, info):
    assert 'distance_to_target' in state and 'target_x' in info
    return 1 - clip_and_norm(state['distance_to_target'], 0, info['target_x'])


def comfort_higher_speed_potential(state, info):
    assert 'ego_vx' in state
    assert 'speed_lower_bound' in info and 'speed_upper_bound' in info
    return 1-clip_and_norm(abs(info['speed_upper_bound'] - state['ego_vx']), info['speed_tol'], info['speed_upper_bound'] - info['speed_tol'])


def comfort_right_lane_potential(state, info):
    assert 'ego_y' in state
    assert 'target_lane_y' in info and 'target_lane_tol' in info and 'max_y' in info
    return 1-clip_and_norm(abs(state['ego_y']-info['target_lane_y']), info['target_lane_tol'], info['max_y']-info['target_lane_y'])


def simple_base_reward(state, info):
    assert 'distance_to_target' in state and 'target_distance_tol' in info
    reached_target = bool(state['distance_to_target'] <= info['target_distance_tol'])
    return 1 if reached_target else 0


class HighwayHierarchicalPotentialShaping(RewardFunction):


    def _safety_potential(self, state, info):
        safety_reward = safety_no_collision_potential(state, info)
        return safety_reward

    def _target_potential(self, state, info):
        target_reward = target_reach_potential(state, info)
        # hierarchical weights
        safety_weight = safety_no_collision_potential(state, info)
        return safety_weight * target_reward

    def _comfort_potential(self, state, info):
        c1 = comfort_higher_speed_potential(state, info)
        c2 = comfort_right_lane_potential(state, info)
        comfort_reward = c1 + c2
        # hierarchical weights
        safety_weight = safety_no_collision_potential(state, info)
        target_weight = target_reach_potential(state, info)
        return safety_weight * target_weight * comfort_reward

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # hierarchical shaping
        shaping_safety = gamma * self._safety_potential(next_state, info) - self._safety_potential(state, info)
        shaping_target = gamma * self._target_potential(next_state, info) - self._target_potential(state, info)
        shaping_comfort = gamma * self._comfort_potential(next_state, info) - self._comfort_potential(state, info)

        return base_reward + shaping_safety + shaping_target + shaping_comfort


class HighwayScalarizedMultiObjectivization(RewardFunction):

    def __init__(self, weights: List[float], **kwargs):
        assert len(weights) == len(get_all_specs()), f"nr weights ({len(weights)}) != nr reqs {len(get_all_specs())}"
        assert (sum(weights) - 1.0) <= 0.0001, f"sum of weights ({sum(weights)}) != 1.0"
        self._weights = weights

    def __call__(self, state, action=None, next_state=None, info=None) -> float:
        base_reward = simple_base_reward(next_state, info)
        if info["done"]:
            return base_reward
        # evaluate individual shaping functions
        shaping_safety = gamma * safety_no_collision_potential(next_state, info) - safety_no_collision_potential(state, info)
        shaping_target = gamma * target_reach_potential(next_state, info) - target_reach_potential(state, info)
        shaping_highspeed = gamma * comfort_higher_speed_potential(next_state, info) - comfort_higher_speed_potential(state, info)
        shaping_rightlane = gamma * comfort_right_lane_potential(next_state, info) - comfort_right_lane_potential(state, info)
        # linear scalarization of the multi-objectivized requirements
        reward = base_reward
        for w, f in zip(self._weights, [shaping_safety, shaping_target, shaping_highspeed, shaping_rightlane]):
            reward += w * f
        return reward


class HighwayUniformScalarizedMultiObjectivization(HighwayScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        weights = np.array([1.0, 1.0, 1.0, 1.0])
        weights /= np.sum(weights)
        super(HighwayUniformScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)


class HighwayDecreasingScalarizedMultiObjectivization(HighwayScalarizedMultiObjectivization):

    def __init__(self, **kwargs):
        """
        weights selected considering a budget of 1.0 + 0.5 + 0.25 + 0.25 = 2.0, then:
            - the sum of safety weights is ~ 1.0/2.0
            - the sum of target weights is ~ 0.50/2.0
            - the sum of comfort weights is ~ 0.50/2.0
        """
        weights = np.array([1.0, 0.5, 0.25, 0.25])
        weights /= np.sum(weights)
        super(HighwayDecreasingScalarizedMultiObjectivization, self).__init__(weights=weights, **kwargs)

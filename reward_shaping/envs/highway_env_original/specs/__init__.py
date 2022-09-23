from reward_shaping.monitor.formula import Operator
import numpy as np

_registry = {}


def get_spec(name):
    return _registry[name]


def get_all_specs():
    return _registry


def register_spec(name, operator, build_predicate):
    if name not in _registry.keys():
        _registry[name] = (operator, build_predicate)


def _build_no_collision(_):
    return lambda state, info: -1 if (state['collision'] == 1) else +1


def _build_high_speed(_):
    return lambda state, info: info['speed_tol'] - abs(info['speed_upper_bound'] - state['ego_vx'])


def _build_right_lane(_):
    return lambda state, info: info['target_lane_tol'] - abs(state['ego_y'] - info['target_lane_y'])


def _build_reach_target(_):
    return lambda state, info: 1 if (state['distance_to_target'] <= info['target_distance_tol']) else -1


register_spec("s1_nocollision", Operator.ENSURE, _build_no_collision)
register_spec("t_origin", Operator.ACHIEVE, _build_reach_target)
register_spec("c1_highspeed", Operator.ENCOURAGE, _build_right_lane)
register_spec("c2_rightlane", Operator.ENCOURAGE, _build_reach_target)

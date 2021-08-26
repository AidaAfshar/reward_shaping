import gym

from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.cart_pole_obst.rewards.baselines import SparseReward, ContinuousReward, WeightedReward
from reward_shaping.envs.cart_pole_obst.rewards.graph_based import GraphWithContinuousScoreBinaryIndicator, \
    GraphWithContinuousScoreContinuousIndicator, GraphWithProgressScoreBinaryIndicator, \
    GraphWithBinarySafetyScoreBinaryIndicator, GraphWithSingleConjunctiveSafetyNode
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import STLReward, BoolSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward

# Baselines
register_reward('default', reward=DefaultReward)
register_reward('sparse', reward=SparseReward)
register_reward('continuous', reward=ContinuousReward)
register_reward('stl', reward=STLReward)
register_reward('bool_stl', reward=BoolSTLReward)
register_reward('weighted', reward=WeightedReward)
# Graph-based (gb) formulations
register_reward('gb_cr_bi', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('gb_cr_ci', reward=GraphWithContinuousScoreContinuousIndicator)
# Graph-based with target score measuring progress (ie, closeness to target w.r.t. the prev step)
register_reward('gb_pcr_bi', reward=GraphWithProgressScoreBinaryIndicator)
# Graph-based with binary score only for safety nodes
register_reward('gb_bcr_bi', reward=GraphWithBinarySafetyScoreBinaryIndicator)
# Graph-based with 1 single safety node (AND_{collision, falldown, outside}
register_reward('gb_cr_bi_s1', reward=GraphWithSingleConjunctiveSafetyNode)

"""
register_reward('hier_cont', reward=GraphWithContinuousScore)
register_reward('hier_cont_pot', reward=PotentialGraphWithContinuousScore)
register_reward('hier_disc', reward=GraphWithContinuousTargetAndDiscreteSafety)
register_reward('hier_disc_pot', reward=PotentialGraphWithContinuousTargetAndDiscreteSafety)
register_reward('hier_binary_ind', reward=GraphWithContinuousScoreBinaryIndicator)
register_reward('hier_cont_ind', reward=GraphWithContinuousScoreContinuousIndicator)
"""
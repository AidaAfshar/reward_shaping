from reward_shaping.core.helper_fns import DefaultReward
from reward_shaping.envs.cart_pole_obst.rewards.baselines import CPOSparseReward, CPOContinuousReward, \
    CPOWeightedBaselineReward, CPOEvalConfig
from reward_shaping.envs.cart_pole_obst.rewards.graph_based import CPOGraphWithContinuousScoreBinaryIndicator, \
    CPOGraphWithContinuousScoreContinuousIndicator, CPOGraphWithProgressScoreBinaryIndicator, \
    CPOGraphWithBinarySafetyScoreBinaryIndicator, CPOChainGraph, CPOGraphBinarySafetyProgressTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressDistanceTargetContinuousIndicator, \
    CPOGraphContinuousSafetyProgressMaxTargetContinuousIndicator, \
    CPOGraphBinarySafetyProgressDistanceTargetContinuousIndicator
from reward_shaping.envs.cart_pole_obst.rewards.stl_based import CPOSTLReward

_registry = {}


def get_reward(name: str):
    return _registry[name]


def register_reward(name: str, reward):
    if name not in _registry.keys():
        _registry[name] = reward


# Baselines
register_reward('stl', reward=CPOSTLReward)
register_reward('eval', reward=CPOEvalConfig)
register_reward('weighted', reward=CPOWeightedBaselineReward)
register_reward('sparse', reward=CPOSparseReward)
register_reward('gb_chain', reward=CPOChainGraph)
# Graph-based with binary safety score, progress target score, continuous sat indicators
register_reward('gb_bpr_ci', reward=CPOGraphBinarySafetyProgressTargetContinuousIndicator)
register_reward('gb_bpr_bi', reward=CPOGraphWithBinarySafetyScoreBinaryIndicator)
register_reward('gb_bpdr_ci', reward=CPOGraphBinarySafetyProgressDistanceTargetContinuousIndicator)

# Continuous Safety Reward (not working well)
register_reward('gb_cpr_ci', reward=CPOGraphContinuousSafetyProgressTargetContinuousIndicator)
register_reward('gb_cpdr_ci',
                reward=CPOGraphContinuousSafetyProgressDistanceTargetContinuousIndicator)  # from Dejan meeting
register_reward('gb_cpmr_ci',
                reward=CPOGraphContinuousSafetyProgressMaxTargetContinuousIndicator)  # target: max(bsat,progr)
# Graph-based with binary score only for safety nodes (THIS IS BEFORE THE UNIFIED APPROACH PROGRESS-BASED)

register_reward('default', reward=DefaultReward)
# Graph-based (gb) formulations
register_reward('gb_cr_bi', reward=CPOGraphWithContinuousScoreBinaryIndicator)
register_reward('gb_cr_ci', reward=CPOGraphWithContinuousScoreContinuousIndicator)
# Graph-based with target score measuring progress (ie, closeness to target w.r.t. the prev step)
register_reward('gb_pcr_bi', reward=CPOGraphWithProgressScoreBinaryIndicator)
# Graph-based with 1 node for each level, evaluated as conjunction (eg, AND_{collision, falldown, outside})
register_reward('continuous', reward=CPOContinuousReward)

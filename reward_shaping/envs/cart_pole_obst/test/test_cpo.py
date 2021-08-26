import warnings
from unittest import TestCase

from reward_shaping.envs.cart_pole_obst.test.test import generic_env_test

env_name = "cart_pole_obst"


class TestCartPoleObstacle(TestCase):

    def test_fixedheight_default(self):
        task = "fixed_height"
        reward = "default"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_sparse(self):
        task = "fixed_height"
        reward = "sparse"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_continuous(self):
        task = "fixed_height"
        reward = "continuous"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_stl(self):
        task = "fixed_height"
        reward = "stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_fixedheight_boolstl(self):
        task = "fixed_height"
        reward = "bool_stl"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_potential(self):
        task = "fixed_height"
        reward = "gb_cr_bi"
        result = generic_env_test(env_name, task, reward, potential=True)
        self.assertTrue(result)
        warnings.warn("the implementation of potential formulation is not garanteed")

    def test_gb_progress(self):
        task = "fixed_height"
        reward = "gb_pcr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_gb_boolsafety(self):
        task = "fixed_height"
        reward = "gb_bcr_bi"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

    def test_weighted_reward(self):
        task = "fixed_height"
        reward = "weighted"
        result = generic_env_test(env_name, task, reward)
        self.assertTrue(result)

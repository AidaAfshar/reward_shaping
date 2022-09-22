from typing import Dict, Text

import gym
import numpy as np

from reward_shaping.envs.highway_env_RSS.env_backend import utils
from reward_shaping.envs.highway_env_RSS.env_backend.env.abstract import AbstractEnv
from reward_shaping.envs.highway_env_RSS.env_backend.env.action import Action
from reward_shaping.envs.highway_env_RSS.env_backend.env.observation import observation_factory
from reward_shaping.envs.highway_env_RSS.env_backend.road.road import Road, RoadNetwork
from reward_shaping.envs.highway_env_RSS.env_backend.utils import near_split
from reward_shaping.envs.highway_env_RSS.env_backend.vehicle.controller import ControlledVehicle
from reward_shaping.envs.highway_env_RSS.env_backend.vehicle.kinematics import Vehicle
from reward_shaping.envs.highway_env_RSS import highway_utils

from gym.spaces import Box


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
            # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
            # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,  # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,
            "normalize_reward": False,
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                # speed=21,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.vehicle.crashed or \
               self.steps >= self.config["duration"] or \
               (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


class HighwayEnvHPRS(HighwayEnvFast):
    """
    A variant of highway-v0 designed for HPRS:
    """

    _params = None

    def __init__(self, **params):
        self._params = params
        super(HighwayEnvHPRS, self).__init__()

        self.time = 0
        self.max_steps = self.config['duration']
        self.ego_avg_speed = []

        self.observation_space = gym.spaces.Dict(dict(
            observation=observation_factory(self, self.config["observation"]).space(),
            ego_x=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_y=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_vx=Box(low=0.0, high=np.inf, shape=(1,)),
            ego_vy=Box(low=-np.inf, high=np.inf, shape=(1,)),
            ego_lane_index=Box(low=0, high=np.inf, shape=(1,)),
            collision=Box(low=0.0, high=np.inf, shape=(1,)),
            road_progress=Box(low=0.0, high=np.inf, shape=(1,)),
            distance_to_target=Box(low=0.0, high=np.inf, shape=(1,)),
            time=Box(low=0.0, high=np.inf, shape=(1,))
        ))

    def default_config(self) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
            "observation": {
                "type": "Kinematics",
                "features": ["presence", "x", "y", "vx", "vy"],
                "features_range": {
                    "x": [-self._params["x_limit"], self._params["x_limit"]],
                    "y": [-self._params["y_limit"], self._params["y_limit"]],
                    "vx": [-self._params["vx_limit"], self._params["vx_limit"]],
                    "vy": [-self._params["vy_limit"], self._params["vy_limit"]]
                }
            },
            "reward_speed_range": [self._params['speed_lower_bound'], self._params['speed_upper_bound']]
        })

        return cfg


    def reached_target(self, obs, info):
        assert len(obs) > 0 and len(obs[0]) == 5
        assert 'target_x' in info and 'target_distance_tol' in info
        ego_x = obs[0][1]
        check_goal = bool(abs(ego_x - info['target_x']) <= info['target_distance_tol'])
        return 1 if check_goal else 0

    def get_ego_road_progress(self, obs, info):
        assert (len(obs) > 0) and (len(obs[0]) == 5)
        ego_obs = obs[0]
        ego_driven_distance = ego_obs[1]  # x
        return ego_driven_distance

    def get_distance_to_target(self, obs, info):
        assert 'target_x' in info
        ego_driven_distance = self.get_ego_road_progress(obs, info)
        if ego_driven_distance >= info['target_x']:
            return 0  # the distance is considered 0 if the ego passes the target
        return info['target_x'] - ego_driven_distance

    def set_info_constants(self, info):
        info['target_x'] = self._params['target_x']
        info['target_distance_tol'] = self._params['target_distance_tol']
        info['x_limit'] = self._params['x_limit']
        info['y_limit'] = self._params['y_limit']
        info['vx_limit'] = self._params['vx_limit']
        info['vy_limit'] = self._params['vy_limit']
        info['lanes_count'] = self._params['lanes_count']
        info['max_steps'] = self._params['max_steps']
        info['speed_lower_bound'] = self._params['speed_lower_bound']
        info['speed_upper_bound'] = self._params['speed_upper_bound']
        info['speed_tol'] = self._params['speed_tol']
        return info

    def denormalize_observation(self, normalized_obs):
        obs = []

        # denormalized
        features_range = [[-self._params["x_limit"], self._params["x_limit"]],
                          [-self._params["y_limit"], self._params["y_limit"]],
                          [-self._params["vx_limit"], self._params["vx_limit"]],
                          [-self._params["vy_limit"], self._params["vy_limit"]]]
        for i in range(len(normalized_obs)):
            vehicle_norm_obs = normalized_obs[i]
            vehicle_obs = []
            vehicle_obs.append(0)
            for j in range(len(features_range)):
                feature_range = features_range[j]
                feature = utils.lmap(vehicle_norm_obs[j + 1], [-1, 1], [feature_range[0], feature_range[1]])
                feature = np.clip(feature, feature_range[0], feature_range[1])
                vehicle_obs.append(feature)
            obs.append(vehicle_obs)

        # to absolute
        ego_obs = obs[0]
        absolute_obs = []
        absolute_obs.append(ego_obs)

        for i in range(1, len(obs)):
            vehicle_obs = obs[i]
            absolute_vehicle_obs = [sum(k) for k in zip(ego_obs, vehicle_obs)]
            absolute_obs.append(absolute_vehicle_obs)

        # add presence
        for i in range(len(absolute_obs)):
            absolute_obs[i][0] = normalized_obs[i][0]

        return absolute_obs

    def step(self, action: Action):
        self.time += 1
        normalized_obs, reward, done, info = super(HighwayEnvHPRS, self).step(action)
        # print(normalized_obs)
        obs = self.denormalize_observation(normalized_obs)
        # print(obs)

        self.ego_avg_speed.append(obs[0][3])
        info = self.set_info_constants(info)

        reached_target = self.reached_target(obs, info)
        collision = float(self.vehicle.crashed)
        ego_road_progress = self.get_ego_road_progress(obs, info)
        distance_to_target = self.get_distance_to_target(obs, info)
        ego_lane_index = highway_utils.lane_id(obs[0][2], self._params)

        done = bool(done or reached_target)

        state = {
            "observation": normalized_obs,
            "ego_x": obs[0][1],
            "ego_y": obs[0][2],
            "ego_vx": obs[0][3],
            "ego_vy": obs[0][4],
            "ego_lane_index": ego_lane_index,
            "collision": collision,
            "road_progress": ego_road_progress,
            "distance_to_target": distance_to_target,
            "time": self.time,
        }

        info["done"] = done
        info["default_reward"] = reward

        # if info['done']:
        #     print(self.time)
        #     print(collision)
        #     print(state['road_progress'])
        #     print(state['ego_lane_index'])
        #     print(reached_target)
        #     print('_________________')

        return state, reward, done, info

    def reset(self):
        obs = super(HighwayEnvHPRS, self).reset()

        self.time = 0
        self.ego_avg_speed = []
        state = {
            "observation": obs,
            "ego_x": 0,
            "ego_y": 0,
            "ego_vx": 0,
            "ego_vy": 0,
            "ego_lane_index": 0,
            "collision": 0,
            "road_progress": 0,
            "distance_to_target": self._params["target_x"],
            "time": self.time
        }
        return state

    def render(self, mode: str = 'human', info={'reward': None, 'return': None}):
        return super(HighwayEnvHPRS, self).render(mode)

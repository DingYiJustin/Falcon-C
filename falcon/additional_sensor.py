#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
from gym import spaces

# from habitat.core.logging import logger
from habitat import VectorEnv, logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.tasks.nav.nav import PointGoalSensor
from hydra.core.config_store import ConfigStore
import habitat_sim

from dataclasses import dataclass
from habitat.config.default_structured_configs import LabSensorConfig
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff

if TYPE_CHECKING:
    from omegaconf import DictConfig

from habitat.tasks.rearrange.utils import UsesArticulatedAgentInterface, get_angle_to_pos

from extensions.obstacle_map import ObstacleMap
from extensions.geometry_utils import xyz_yaw_to_tf_matrix
from depth_camera_filtering import filter_depth

import cv2

from habitat.utils.visualizations import fog_of_war, maps
import copy

@dataclass
class OracleShortestPathSensorConfig(LabSensorConfig):
    
    type: str = "OracleShortestPathSensor"

@dataclass
class OracleFollowerSensorConfig(LabSensorConfig):
    
    type: str = "OracleFollowerSensor"


@dataclass
class HumanVelocitySensorConfig(LabSensorConfig):
    type: str = "HumanVelocitySensor"

@dataclass
class HumanNumSensorConfig(LabSensorConfig):
    type: str = "HumanNumSensor"
    max_num: int = 6

@dataclass
class RiskSensorConfig(LabSensorConfig):
    type: str = "RiskSensor"
    thres: float = 3.0
    use_geo_distance: bool = True

@dataclass
class SocialCompassSensorConfig(LabSensorConfig):
    type: str = "SocialCompassSensor"
    thres: float = 9.0
    num_bins: int = 8

@dataclass
class OracleHumanoidFutureTrajectorySensorConfig(LabSensorConfig):
    type: str = "OracleHumanoidFutureTrajectorySensor"
    future_step: int = 8 #5

@dataclass
class OracleHumanoidFutureTrajectoryMapSensorConfig(LabSensorConfig):
    type: str = "OracleHumanoidFutureTrajectoryMapSensor"
    future_step: int = 5
    
@dataclass
class TopDownMapSensorConfig(LabSensorConfig):
    type: str = "TopDownMapSensor"
    
@dataclass
class TopDownMapWithTrajectorySensorConfig(LabSensorConfig):
    type: str = "TopDownMapWithTrajectorySensor"

@dataclass
class TopDownMapWithHumanSensorConfig(LabSensorConfig):
    type: str = "TopDownMapWithHumanSensor"
    future_step: int = 8 # 5

@dataclass
class MPCWithHumanSensorConfig(LabSensorConfig):
    type: str = "MPCWithHumanSensor"
    future_step: int =  5
    
@dataclass
class MPCWithHumanMapSensorConfig(LabSensorConfig):
    type: str = "MPCWithHumanMapSensor"
    future_step: int =  5


@registry.register_sensor(name="OracleShortestPathSensor")
class OracleShortestPathSensor(Sensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_shortest_path_sensor"

    def __init__(
        self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,3)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_2(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points[:2] if found_path else [point_a, point_b]
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = np.array(agent_state.position, dtype=np.float32)
        # rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # return [agent_position, goal_position]
        return self._path_to_point_2(
            agent_position, goal_position
        )

@registry.register_sensor(name="OracleFollowerSensor")
class OracleFollowerSensor(PointGoalSensor):
    r"""Sensor that used for A* and ORCA
    """
    cls_uuid: str = "oracle_follower_sensor"
        
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )
    
    def _path_to_point_1(self, point_a, point_b):
        """Get the shortest path between two points."""
        path = habitat_sim.ShortestPath()  # habitat_sim
        path.requested_start = point_a 
        path.requested_end = point_b
        found_path = self._sim.pathfinder.find_path(path)
        return path.points[1] if found_path else [point_b]
    
    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, self._path_to_point_1(agent_position,goal_position)
        )

@registry.register_sensor
class HumanVelocitySensor(UsesArticulatedAgentInterface, Sensor):
    """
    The position and angle of the articulated_agent in world coordinates.
    """

    cls_uuid = "human_velocity_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim
        self.value = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6, dtype=np.float64)

    def _get_uuid(self, *args, **kwargs):
        return HumanVelocitySensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(6,6),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        # human_num = kwargs["task"]._human_num
        for i in range(self._sim.num_articulated_agents-1):
            articulated_agent = self._sim.get_agent_data(i+1).articulated_agent
            human_pos = np.array(articulated_agent.base_pos, dtype=np.float64)
            human_rot = np.array([float(articulated_agent.base_rot)], dtype=np.float64)
            human_vel = np.array(kwargs['task'].measurements.measures['human_velocity_measure']._metric[i],dtype=np.float64)
            self.value[i] = np.concatenate((human_pos, human_rot, human_vel))
        return self.value
    
@registry.register_sensor
class HumanNumSensor(UsesArticulatedAgentInterface, Sensor):
    """
    The num of the other agent in world.
    (in our setup, agents except agent_0 are humanoids)
    """

    cls_uuid = "human_num_sensor"

    def __init__(self, sim, config, *args, **kwargs):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args, **kwargs):
        return HumanNumSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(1,), low=0, high=6, dtype=np.int32
        )

    def get_observation(self, observations, episode, *args, **kwargs):    
        if "human_num" in episode.info:
            human_num = min(episode.info['human_num'], 6)
        else:
            human_num = min(self._sim.num_articulated_agents - 1, 6)
        # Ensure the returned value is a tensor with shape (1,)
        return np.array([human_num], dtype=np.int32)

@registry.register_sensor
class RiskSensor(UsesArticulatedAgentInterface, Sensor):
    r"""Sensor for observing social risk to which the agent is subjected".

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "risk_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        self._robot_idx = 0
        self.thres = config.thres
        self._use_geo_distance = config.use_geo_distance
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def get_observation(
        self, observations, episode, *args, **kwargs
    ):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        if self._human_nums == 0:
            return np.array([0], dtype=np.float32)
        else:
            robot_pos = self._sim.get_agent_state(0).position

            human_pos = []
            human_dis = []

            for i in range(self._human_nums):
                human_position = self._sim.get_agent_state(i+1).position
                human_pos.append(human_position)

                if self._use_geo_distance:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = robot_pos
                    path.requested_end = human_position
                    found_path = self._sim.pathfinder.find_path(path)

                    if found_path:
                        distance = self._sim.geodesic_distance(robot_pos, human_position)
                    else:
                        distance = np.linalg.norm(human_position - robot_pos, ord=2)
                else:
                    distance = np.linalg.norm(human_position - robot_pos, ord=2)

                human_dis.append(distance)

            return np.array([max(1 - min(human_dis) / self.thres, 0)],
                            dtype=np.float32)

@registry.register_sensor
class SocialCompassSensor(UsesArticulatedAgentInterface, Sensor):
    r"""
    Implementation of people relative position sensor
    """

    cls_uuid: str = "social_compass_sensor"

    def __init__(
        self, sim, config, *args, **kwargs
    ):
        self._sim = sim
        # parameters
        self.thres = config.thres
        self.num_bins = config.num_bins
        super().__init__(config=config)

    def _get_uuid(self, *args, **kwargs):
        return self.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(low=0, high=np.inf, shape=(self.num_bins,),
                          dtype=np.float32)

    def get_polar_angle(self, agent_id = 0):
        agent_state = self._sim.get_agent_state(agent_id)
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
    
    def get_heading_error(self, source, target):
        r"""Computes the difference between two headings (radians); can be negative
        or positive.
        """
        diff = target - source
        if diff > np.pi:
            diff -= np.pi*2
        elif diff < -np.pi:
            diff += np.pi*2
        return diff
    
    def get_observation(self, observations, episode, *args, **kwargs):
        self._human_nums = min(episode.info['human_num'], self._sim.num_articulated_agents - 1)
        angles = [0] * self.num_bins
        if self._human_nums == 0:
            return np.array(angles, dtype=np.float32)
        else:
            a_pos = self._sim.get_agent_state(0).position
            a_head = self._sim.get_agent_state(0).rotation  # 2*np.arccos(self._sim.get_agent_state().rotation.w)

            a_head = -self.get_polar_angle(0) + np.pi / 2  # -quat_to_rad(a_head) + np.pi / 2

            for i in range(self._human_nums):
                pos = self._sim.get_agent_state(i+1).position
                theta = math.atan2(pos[2] - a_pos[2], pos[0] - a_pos[0])
                theta = self.get_heading_error(a_head, theta)
                theta = theta if theta > 0 else 2 * np.pi + theta

                bin = int(theta / (2 * np.pi / self.num_bins))

                dist = np.sqrt((pos[2] - a_pos[2]) ** 2 + (pos[0] - a_pos[
                    0]) ** 2)  # self._sim.geodesic_distance(a_pos, pos)
                norm_dist = max(1 - dist / self.thres, 0)
                if norm_dist > angles[bin]:
                    angles[bin] = norm_dist

            return np.array(angles, dtype=np.float32)

@registry.register_sensor
class OracleHumanoidFutureTrajectorySensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "oracle_humanoid_future_trajectory"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        self.future_step = kwargs['config']['future_step'] 
        self.max_human_num = 6
        self.human_num = task._human_num
        self.result_list = None  

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OracleHumanoidFutureTrajectorySensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self.max_human_num, self.future_step, 2),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_result_list(human_num, future_step, max_human_num):
        """Initialize the result list with default values."""
        result = np.full((max_human_num, future_step, 2), -100, dtype=np.float32)
        return result

    def get_observation(self, task, *args, **kwargs):
        human_num = self._task._human_num

        if self.result_list is None or human_num != self.human_num:
            self.result_list = self._initialize_result_list(human_num, self.future_step, self.max_human_num)
            self.human_num = human_num
        
        if self.human_num == 0:
            return self.result_list
        
        human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
        if not human_future_trajectory:
            return self.result_list

        robot_pos = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos)[[0, 2]]

        for key, trajectories in human_future_trajectory.items():
            trajectories = np.array(trajectories)
            trajectories = trajectories.astype('float32')
            self.result_list[key - 1, :len(trajectories), :] = (trajectories[:, [0, 2]] - robot_pos)

        return self.result_list.tolist()

@registry.register_sensor
class OracleHumanoidFutureTrajectoryMapSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "oracle_humanoid_future_trajectory_map"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        self.future_step = kwargs['config']['future_step'] 
        # self.max_human_num = 6
        self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return OracleHumanoidFutureTrajectoryMapSensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(101,101, self.future_step),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((self.future_step, 101, 101), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        try:
            human_num = self._task._human_num
            future_map = self._initialize_map(self)
            # Initialize the map instead of the result list
            if human_num != self.human_num:
                self.human_num = human_num

            if self.human_num == 0:
                return np.transpose(future_map, (1, 2, 0))
            
            human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
            if not human_future_trajectory:
                return np.transpose(future_map, (1, 2, 0))

            # Get robot position
            robot_pos = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos)[[0, 2]]
            
            # Centering the robot position at the center of the map
            # robot_pixel_x = 50  # Center x in a 101x101 map
            # robot_pixel_y = 50  # Center y in a 101x101 map

            for key, trajectories in human_future_trajectory.items():
                trajectories = np.array(trajectories)
                trajectories = trajectories.astype('float32')
                
                for t in range(len(trajectories)):
                    human_position = trajectories[t, [0, 2]] - robot_pos
                    
                    # Calculate distance from robot to human
                    distance = np.linalg.norm(human_position)

                    # Only draw if the human is within 10 meters
                    if distance <= 10.0:
                        # Convert human position to pixel coordinates
                        human_pixel_x = int(50 + (-human_position[1]) * 10)  # Centering the human position
                        human_pixel_y = int(50 + (-human_position[0]) * 10)  # Centering the human position
                        
                        # Ensure pixel coordinates are within the map bounds
                        if 0 <= human_pixel_x < 101 and 0 <= human_pixel_y < 101 and t<self.future_step:
                            # Draw a filled circle for the human on the map
                            # future_map = np.ascontiguousarray(future_map)
                            cv2.circle(future_map[t, :, :], (human_pixel_x, human_pixel_y), radius=3, color=1, thickness=-1)
                # cv2.circle(future_map[t, :, :], (50, 50), radius=2, color=1, thickness=-1)
            future_map = np.transpose(future_map, (1, 2, 0))
            # print('future_map.shape',future_map.shape)
            # if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
            #     self._obstacle_map = ObstacleMap(min_height=0.61, max_height=0.88, agent_radius=0.25)
            #     self.current_episode_id = episode.episode_id
            #     self.current_episode_scene_id = episode.scene_id
            
            agent_state = self._sim.get_agent_state()

            rotation_world_start = quaternion_from_coeff(episode.start_rotation)
            rotation_world_agent = agent_state.rotation
            
            camera_yaw = self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )[0] #- to rotate back
            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                self.current_episode_init_yaw = camera_yaw
                # print('self.current_episode_init_yaw', camera_yaw)
            
            camera_yaw -= self.current_episode_init_yaw
            # print('camera_yaw degree ', np.degrees(camera_yaw))
            
            center = (future_map.shape[0] // 2, future_map.shape[1] // 2)  # (width, height) for OpenCV
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(camera_yaw), 1.0) 
            for i in range(self.future_step):
                future_map[:,:,i] = cv2.warpAffine(future_map[:,:,i], rotation_matrix, (101, 101), flags=cv2.INTER_LINEAR)
            return future_map
        
        except Exception as e:
            print("Human Future Trajectory Map Sensor Exception", e)
            return np.zeros((101, 101, self.future_step), dtype=np.float32)
            

@registry.register_sensor
class TopDownMapSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "td_map"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        # self.max_human_num = 6
        # self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None
        
        self._top_down_map = None

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return TopDownMapSensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(101,101, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((3, 101, 101), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        try:
            future_map = self._initialize_map(self)
            agent_state = self._sim.get_agent_state()

            rotation_world_start = quaternion_from_coeff(episode.start_rotation)
            rotation_world_agent = agent_state.rotation

            camera_yaw = self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )[0] #- to rotate back
            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                self.current_episode_init_yaw = camera_yaw
                self._top_down_map = maps.get_topdown_map_from_sim(
                    self._sim,
                    map_resolution=1024,#self._map_resolution,
                    draw_border= True, #self._config.draw_border,
                    meters_per_pixel = 0.1
                )
                # print('self.current_episode_init_yaw', camera_yaw)
            # Initialize the map instead of the result list
            # robot_pos = self._sim.get_agent_data(0).articulated_agent.base_pos
            robot_pos = self._sim.get_agent_state(0).position

            robot_pos = maps.to_grid(
                robot_pos[2],
                robot_pos[0],
                (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                sim=self._sim,
            ) #a_x, a_y 
            
            # Centering the robot position at the center of the map
            robot_pixel_x = 50  # Center x in a 101x101 map
            robot_pixel_y = 50  # Center y in a 101x101 map

            start_row = robot_pos[0] - robot_pixel_x
            start_col = robot_pos[1] - robot_pixel_y

            # Step 5: Fill the smaller map with the relevant portion of the larger map
            for i in range(101):
                for j in range(101):
                    large_row = start_row + i
                    large_col = start_col + j
                    
                    # Check if within bounds of the large map
                    if 0 <= large_row < self._top_down_map.shape[0] and 0 <= large_col < self._top_down_map.shape[1]:
                        # future_map[self._top_down_map[large_row, large_col], i, j] = 1   
                        future_map[self._top_down_map[large_row, large_col], i, j] = 1   
            future_map = np.transpose(future_map, (1, 2, 0))


            # camera_yaw -= self.current_episode_init_yaw
            # # print('camera_yaw degree ', np.degrees(camera_yaw))
            camera_yaw -= self.current_episode_init_yaw
            camera_yaw = -camera_yaw
            
            center = (future_map.shape[0] // 2, future_map.shape[1] // 2)  # (width, height) for OpenCV
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(camera_yaw)+90, 1.0) 
            for i in range(3):
                future_map[:,:,i] = cv2.warpAffine(future_map[:,:,i], rotation_matrix, (101, 101), flags=cv2.INTER_LINEAR)
            future_map = future_map[:, ::-1,:]

            return future_map
        
        except Exception as e:
            logger.info("Top Down Map Sensor Exception", e)
            return np.zeros((101, 101, 5), dtype=np.float32)
        

@registry.register_sensor
class TopDownMapWithTrajectorySensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "td_map_with_traj"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        # self.max_human_num = 6
        # self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None
        
        self._top_down_map = None
        
        
        self.map_size = 101
        self.agent_diameter_pixels = 5
        self.agent_radius_pixels = self.agent_diameter_pixels // 2
        self.step_distance_pixels = 2
        self.angle_step_radians = 30/120 # 20/120  #10 / 120
        
        # Define the center position
        self.center_position = (self.map_size // 2, self.map_size // 2)

        # Actions
        self.actions = ['left', 'right', 'forward']

        # Initialize agent direction (facing downwards)
        self.agent_angle = 3 * math.pi / 2  # 270 degrees
        
        self.future_step = 8 #5

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return TopDownMapWithTrajectorySensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(101,101, 3),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((1, 101, 101), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def is_collision(self, position, occupancy_map, agent_radius):
        # Check if the agent's area overlaps with any obstacles
        x, y = position
        for i in range(-agent_radius, agent_radius + 1):
            for j in range(-agent_radius, agent_radius + 1):
                if occupancy_map[min(max(y + i, 0), self.map_size-1), min(max(x + j, 0), self.map_size-1),0 ] == 1:
                    return True
        return False

    def move_agent(self,occupancy_map , position, angle, action):
        precise_x, precise_y = position
        x, y = round(precise_x), round(precise_x)
        if action == 'left':
            angle += self.angle_step_radians
        elif action == 'right':
            angle -= self.angle_step_radians
        elif action == 'forward':
            # Calculate potential new position
            precise_x = precise_x + self.step_distance_pixels * math.cos(angle)
            precise_y = precise_y - self.step_distance_pixels * math.sin(angle)
            new_x = round(precise_x)
            new_y = round(precise_y)
            x, y = new_x, new_y
            
            # Check for collision
            # if not self.is_collision((new_x, new_y), occupancy_map, self.agent_radius_pixels):
            #     x, y = new_x, new_y
            # else:
            #     # Stay close to the obstacle; calculate the closest valid position
            #     last_valid_position = (x, y) #自己加的
            #     i = 0
            #     while not self.is_collision((x, y), occupancy_map, self.agent_radius_pixels) and i<25:
            #         last_valid_position = (x, y)
            #         x = int(x + 1/120 * math.cos(angle))
            #         y = int(y - 1/120 *math.sin(angle))
            #         i+=1
            #     x, y = last_valid_position

        return (x, y), (precise_x, precise_y ), angle

    def draw_trajectory(self, map_img, occupancy_map, step_colors):
        position = self.center_position
        angle = self.agent_angle
        previous_position = position
        precise_position = position
        
        currentTimeStep = [(position,precise_position , angle)]
        nextTimeStep = []
        # for step, action in enumerate(trajectory):
        for step in range(self.future_step):
            while len(currentTimeStep) > 0:
                cposition,cprecise_position , cangle = currentTimeStep.pop()
                for action in self.actions:
                    previous_position = cposition
                    position,precise_position , angle = self.move_agent(occupancy_map, cprecise_position, cangle, action)
                    nextTimeStep.append((position,precise_position , angle))
                    # Draw the path from the previous position to the current position with a thickness equal to the agent's diameter
                    if position != previous_position:
                        cv2.line(map_img, previous_position, position, step_colors[step], self.agent_diameter_pixels)
                    # Draw the agent's position for this step
                    cv2.circle(map_img, position, self.agent_radius_pixels, step_colors[step], -1)

            currentTimeStep = nextTimeStep
            nextTimeStep = []
                    
            # print('angle end[i]', angle)
        # logger.info('angle start', self.agent_angle,'angle end', angle,'math.cos(angle)', math.cos(angle), 'math.cos(self.agent_angle)', math.cos(self.agent_angle) ,'math.sin(angle)',math.sin(angle), 'math.sin(self.agent_angle)', math.sin(self.agent_angle) ,  position, int(position[0] + self.step_distance_pixels * math.cos(angle)), int(position[1] - self.step_distance_pixels * math.sin(angle)))

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        
        try:
            # Initialize agent direction (facing downwards)
            self.agent_angle = 3 * math.pi / 2  # 270 degrees

            future_map = self._initialize_map(self)
            agent_state = self._sim.get_agent_state()

            rotation_world_start = quaternion_from_coeff(episode.start_rotation)
            rotation_world_agent = agent_state.rotation

            camera_yaw = self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )[0] #- to rotate back
            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                self.current_episode_init_yaw = camera_yaw
                self._top_down_map = maps.get_topdown_map_from_sim(
                    self._sim,
                    map_resolution=1024,#self._map_resolution,
                    draw_border= True, #self._config.draw_border,
                    meters_per_pixel = 0.1
                )
                # print('self.current_episode_init_yaw', camera_yaw)
            # Initialize the map instead of the result list
            # robot_pos = self._sim.get_agent_data(0).articulated_agent.base_pos
            robot_pos = self._sim.get_agent_state(0).position

            robot_pos = maps.to_grid(
                robot_pos[2],
                robot_pos[0],
                (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                sim=self._sim,
            ) #a_x, a_y 
            
            # Centering the robot position at the center of the map
            robot_pixel_x = 50  # Center x in a 101x101 map
            robot_pixel_y = 50  # Center y in a 101x101 map

            start_row = robot_pos[0] - robot_pixel_x
            start_col = robot_pos[1] - robot_pixel_y

            # Step 5: Fill the smaller map with the relevant portion of the larger map
            for i in range(101):
                for j in range(101):
                    large_row = start_row + i
                    large_col = start_col + j
                    
                    # Check if within bounds of the large map
                    if 0 <= large_row < self._top_down_map.shape[0] and 0 <= large_col < self._top_down_map.shape[1]:
                        # future_map[self._top_down_map[large_row, large_col], i, j] = 1   
                        if self._top_down_map[large_row, large_col] ==  0:
                            future_map[0, i, j] = 1   
            future_map = np.transpose(future_map, (1, 2, 0))


            # camera_yaw -= self.current_episode_init_yaw
            # # print('camera_yaw degree ', np.degrees(camera_yaw))
            camera_yaw -= self.current_episode_init_yaw
            camera_yaw = -camera_yaw
            
            center = (future_map.shape[0] // 2, future_map.shape[1] // 2)  # (width, height) for OpenCV
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(camera_yaw)+90, 1.0) 
            for i in range(1):
                future_map[:,:,i] = cv2.warpAffine(future_map[:,:,i], rotation_matrix, (101, 101), flags=cv2.INTER_LINEAR)
            future_map = future_map[:, ::-1,:]
            
            color_map = cv2.cvtColor(future_map * 255, cv2.COLOR_GRAY2BGR)
            
            # trajectories = [('left','left','left', 'forward','forward')]
            # trajectories = [(a1, a2, a3, a4, a5) for a1 in self.actions for a2 in self.actions for a3 in self.actions 
            #         for a4 in self.actions for a5 in self.actions] # for a6 in self.actions for a7 in self.actions for a8 in self.actions]
            step_colors = [
                (255, 0, 0),   # Red
                (0, 255, 0),   # Green
                (0, 0, 255),   # Blue
                (255, 255, 0), # Cyan
                (255, 0, 255), # Magenta
                (0, 255, 255), # Yellow
                (128, 0, 128), # Purple
                (0, 128, 128)  # Teal
            ]

            # for trajectory in trajectories:
            self.draw_trajectory(color_map,future_map , step_colors)

            return color_map
        
        except Exception as e:
            logger.info("Top Down Map with Trajectory Sensor Exception", e)
            return np.zeros((101, 101, 5), dtype=np.float32)


@registry.register_sensor
class TopDownMapWithHumanSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "td_map_with_human"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        # self.max_human_num = 6
        # self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None
        
        self._top_down_map = None
        
        
        self.map_size = 101
        self.agent_diameter_pixels = 5
        self.agent_radius_pixels = self.agent_diameter_pixels // 2
        self.step_distance_pixels = 2
        self.angle_step_radians = 30/120 #20 / 120 #10 / 120
        
        # Define the center position
        self.center_position = (self.map_size // 2, self.map_size // 2)

        # Actions
        self.actions = ['left', 'right', 'forward']

        # Initialize agent direction (facing downwards)
        self.agent_angle = 3 * math.pi / 2  # 270 degrees
        
        self.future_step = kwargs['config']['future_step'] 
        # self.max_human_num = 6
        self.human_num = task._human_num
        self.episode = None

        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return TopDownMapWithHumanSensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3**self.future_step,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((self.future_step+1, 101, 101), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def is_collision(self, position, occupancy_map, agent_radius):
        # Check if the agent's area overlaps with any obstacles
        x, y = position
        for i in range(-agent_radius, agent_radius + 1):
            for j in range(-agent_radius, agent_radius + 1):
                if occupancy_map[min(max(y + i, 0), self.map_size-1), min(max(x + j, 0), self.map_size-1),0 ] == 1:
                    return True
        return False

    def move_agent(self,occupancy_map , position, angle, action):
        precise_x, precise_y = position
        x, y = round(precise_x), round(precise_x)
        if action == 'left':
            angle += self.angle_step_radians
        elif action == 'right':
            angle -= self.angle_step_radians
        elif action == 'forward':
            # Calculate potential new position
            precise_x = precise_x + self.step_distance_pixels * math.cos(angle)
            precise_y = precise_y - self.step_distance_pixels * math.sin(angle)
            new_x = round(precise_x)
            new_y = round(precise_y)
            
            x, y = new_x, new_y
            # Check for collision
            # if not self.is_collision((new_x, new_y), occupancy_map, self.agent_radius_pixels):
            #     x, y = new_x, new_y
            # else:
            #     # Stay close to the obstacle; calculate the closest valid position
            #     last_valid_position = (x, y) #自己加的
            #     i = 0
            #     while not self.is_collision((x, y), occupancy_map, self.agent_radius_pixels) and i<25:
            #         last_valid_position = (x, y)
            #         x = int(x + 1/120 * math.cos(angle))
            #         y = int(y - 1/120 *math.sin(angle))
            #         i+=1
            #     x, y = last_valid_position

        return (x, y),(precise_x, precise_y ), angle

    def draw_trajectory(self, occupancy_map):
        position = self.center_position
        
        angle = self.agent_angle
        previous_position = position
        precise_position = position
        
        returnV = np.zeros((3**self.future_step,))
        currentTimeStep = [(0, position, precise_position, angle)]
        nextTimeStep = []
        
        # for step, action in enumerate(trajectory):
        for step in range(self.future_step): 
            while len(currentTimeStep)>0:
                cvalue, cposition, cprecise_position, cangle = currentTimeStep.pop()
                for action in self.actions:
                    previous_position = cposition
                    value = cvalue
                    if cvalue == -100:
                        nextTimeStep.append((value, position, precise_position, angle))
                        continue
                    map_img = np.zeros((101, 101), dtype=np.float32)
                    position, precise_position, angle = self.move_agent(occupancy_map, cprecise_position, cangle, action)
                    # Draw the path from the previous position to the current position with a thickness equal to the agent's diameter
                    if position != previous_position:
                        cv2.line(map_img, previous_position, position, 1, self.agent_diameter_pixels)
                    # Draw the agent's position for this step
                    cv2.circle(map_img, position, self.agent_radius_pixels, 1, -1)
                    exists_overlap = np.any(np.logical_and(map_img == 1, occupancy_map[:,:,step+1] > 0.5))
                    if exists_overlap:
                        value = -100
                    nextTimeStep.append((value, position, precise_position, angle))
            currentTimeStep = nextTimeStep
            nextTimeStep = []
            
        for i in range(len(currentTimeStep)):
            returnV[i] = currentTimeStep[i][0]
            
        
        return returnV
        # return distance_to_target1 - distance_to_target

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        
        try:
            # Initialize agent direction (facing downwards)
            self.agent_angle = 3 * math.pi / 2  # 270 degrees

            future_map = self._initialize_map(self)
            agent_state = self._sim.get_agent_state()

            rotation_world_start = quaternion_from_coeff(episode.start_rotation)
            rotation_world_agent = agent_state.rotation

            camera_yaw = self._quat_to_xy_heading(
                rotation_world_agent.inverse() * rotation_world_start
            )[0] #- to rotate back
            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                self.current_episode_init_yaw = camera_yaw
                self._top_down_map = maps.get_topdown_map_from_sim(
                    self._sim,
                    map_resolution=1024,#self._map_resolution,
                    draw_border= True, #self._config.draw_border,
                    meters_per_pixel = 0.1
                )
                self.episode = episode
                # print('self.current_episode_init_yaw', camera_yaw)
            # Initialize the map instead of the result list
            # robot_pos = self._sim.get_agent_data(0).articulated_agent.base_pos
            robot_pos = self._sim.get_agent_state(0).position

            robot_pos = maps.to_grid(
                robot_pos[2],
                robot_pos[0],
                (self._top_down_map.shape[0], self._top_down_map.shape[1]),
                sim=self._sim,
            ) #a_x, a_y 
            
            # Centering the robot position at the center of the map
            robot_pixel_x = 50  # Center x in a 101x101 map
            robot_pixel_y = 50  # Center y in a 101x101 map

            start_row = robot_pos[0] - robot_pixel_x
            start_col = robot_pos[1] - robot_pixel_y

            # Step 5: Fill the smaller map with the relevant portion of the larger map
            index_map = np.zeros((101,101, 2))
            for i in range(101):
                for j in range(101):
                    large_row = start_row + i
                    large_col = start_col + j
                    
                    # Check if within bounds of the large map
                    if 0 <= large_row < self._top_down_map.shape[0] and 0 <= large_col < self._top_down_map.shape[1]:
                        # future_map[self._top_down_map[large_row, large_col], i, j] = 1   
                        if self._top_down_map[large_row, large_col] ==  0:
                            future_map[0, i, j] = 1   
                            index_map[i,j, 0] =large_row
                            index_map[i,j, 1] =large_col
                            
                            
            human_num = self._task._human_num
            # Initialize the map instead of the result list
            if human_num != self.human_num:
                self.human_num = human_num

            # if self.human_num == 0:
            #     return np.transpose(future_map, (1, 2, 0))
            
            human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
            # if not human_future_trajectory:
            #     return np.transpose(future_map, (1, 2, 0))

            # Get robot position
            # robot_pos = np.array(self._sim.get_agent_data(0).articulated_agent.base_pos)[[0, 2]]
            robot_pos = np.array(self._sim.get_agent_state(0).position)[[0, 2]]

            if human_future_trajectory is not None:
                for key, trajectories in human_future_trajectory.items():
                    trajectories = np.array(trajectories)
                    trajectories = trajectories.astype('float32')
                    
                    for t in range(len(trajectories)):
                        human_position = trajectories[t, [0, 2]] - robot_pos
                        
                        # Calculate distance from robot to human
                        distance = np.linalg.norm(human_position)

                        # Only draw if the human is within 10 meters
                        if distance <= 10.0:
                            # Convert human position to pixel coordinates
                            human_pixel_x = int(50 + (-human_position[1]) * 10)  # Centering the human position
                            human_pixel_y = int(50 + (-human_position[0]) * 10)  # Centering the human position
                            
                            # Ensure pixel coordinates are within the map bounds
                            if 0 <= human_pixel_x < 101 and 0 <= human_pixel_y < 101 and t<self.future_step:
                                # Draw a filled circle for the human on the map
                                # future_map = np.ascontiguousarray(future_map)
                                cv2.circle(future_map[t+1, :, :], (human_pixel_x, human_pixel_y), radius=3, color=1, thickness=-1)
                
            
            future_map = np.transpose(future_map, (1, 2, 0))


            # camera_yaw -= self.current_episode_init_yaw
            # # print('camera_yaw degree ', np.degrees(camera_yaw))
            camera_yaw -= self.current_episode_init_yaw
            camera_yaw = -camera_yaw
            
            center = (future_map.shape[0] // 2, future_map.shape[1] // 2)  # (width, height) for OpenCV
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(camera_yaw)+90, 1.0) 
            for i in range(self.future_step+1):
                future_map[:,:,i] = cv2.warpAffine(future_map[:,:,i], rotation_matrix, (101, 101), flags=cv2.INTER_LINEAR)
            future_map[:, :,0] = future_map[:, ::-1,0]
            
            for i in range(2):
                index_map[:,:,i] = cv2.warpAffine(index_map[:,:,i], rotation_matrix, (101, 101), flags=cv2.INTER_NEAREST)
            index_map[:, :,0] = index_map[:, ::-1,0]
            index_map[:, :,1]  = index_map[:, ::-1,1]
            
            
            trajectories = [(a1, a2, a3, a4, a5) for a1 in self.actions for a2 in self.actions for a3 in self.actions 
                    for a4 in self.actions for a5 in self.actions] # for a6 in self.actions for a7 in self.actions for a8 in self.actions]
            # trajectories = []
            
            # traj_costs = np.zeros((3**self.future_step,))
            # for i, trajectory in enumerate(trajectories):
                
            #     traj_costs[i] = self.draw_trajectory(future_map , trajectory, index_map)
            return self.draw_trajectory(future_map )
            # logger.info("Top Down Map with Trajectory Sensor", traj_costs)
            # return traj_costs
        
        except Exception as e:
            logger.info("Top Down Map with Human Sensor Exception", e)
            return np.zeros((101, 101, 5), dtype=np.float32)


@registry.register_sensor
class MPCWithHumanSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "mpc_with_human"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        # self.max_human_num = 6
        # self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None
        
        self._top_down_map = None
        
        
        self.map_size = 101
        self.agent_diameter_pixels = 5
        self.agent_radius_pixels = self.agent_diameter_pixels // 2
        self.step_distance_pixels = 25/120
        self.angle_step_radians = 10/120 #20 / 120 #10 / 120
        
        # Define the center position
        self.center_position = (self.map_size // 2, self.map_size // 2)

        # Actions
        self.actions = ['left', 'right', 'forward']

        # Initialize agent direction (facing downwards)
        self.agent_angle = 3 * math.pi / 2  # 270 degrees
        
        self.future_step = kwargs['config']['future_step'] 
        # self.max_human_num = 6
        self.human_num = task._human_num
        self.episode = None

        self.trajectories = [(a1, a2, a3, a4, a5) for a1 in self.actions for a2 in self.actions for a3 in self.actions 
                    for a4 in self.actions for a5 in self.actions]
        
        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MPCWithHumanSensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3**self.future_step,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((101, 101, 3), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def move_agent(self , position, angle, action):
        precise_x, precise_y = position
        
        if action == 'left':
            angle += self.angle_step_radians
        elif action == 'right':
            angle -= self.angle_step_radians
        elif action == 'forward':
            # Calculate potential new position
            precise_x = precise_x + 25/120 * math.cos(angle)
            precise_y = precise_y - 25/120 * math.sin(angle)

        return (precise_x, precise_y ), angle
        

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        
        try:
            # Initialize agent direction (facing downwards)


            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                # self.current_episode_init_yaw = camera_yaw
                self.episode = episode

            human_num = self._task._human_num
            # Initialize the map instead of the result list
            if human_num != self.human_num:
                self.human_num = human_num


            human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
            returnMap = self._initialize_map(self)
            # logger.info(f"check {precise_position[0]} {precise_position[1]} {heading_angle}")
            
            returnV = np.zeros((3**self.future_step,))
            logger.info(f"check {3**self.future_step} {len(self.trajectories)} {returnV.shape[0]} ")
            for i, trajectory in enumerate(self.trajectories):
                robot_pos = np.array(self._sim.get_agent_state(0).position)[[0, 2]]
                precise_position = (float(robot_pos[0]), float(robot_pos[1]))


                T = self._sim.get_agent_data(0).articulated_agent.base_transformation
                forward = np.array([1.0, 0, 0])
                heading_angle = get_angle_to_pos(T.transform_vector(forward))
                
                for step, action in enumerate(trajectory):
                    px, py = precise_position
                    precise_position, heading_angle = self.move_agent(precise_position, heading_angle, action)
                    probot_pos = (px- robot_pos[0], py- robot_pos[1])
                    crobot_pos = (precise_position[0]- robot_pos[0], precise_position[1]- robot_pos[1])
                    
                    if human_future_trajectory is not None:
                        for key, trajectories in human_future_trajectory.items():
                            trajectories = np.array(trajectories)
                            trajectories = trajectories.astype('float32')
                            
                            human_position = trajectories[step, [0, 2]] - robot_pos
                            if math.sqrt( (crobot_pos[0] - human_position[0])**2 + (crobot_pos[1] - human_position[1])**2) <=0.55:
                                returnV[i] = -100
                                break
                    
            return returnV
        
        except Exception as e:
            logger.info("MPC with Human Sensor Exception", e)
            return np.zeros((101, 101, 5), dtype=np.float32)

@registry.register_sensor
class MPCWithHumanMapSensor(UsesArticulatedAgentInterface, Sensor):
    """
    Assumed Oracle Humanoid Future Trajectory Sensor.
    """

    cls_uuid: str = "mpc_with_human_map"

    def __init__(self, *args, sim, task, **kwargs):
        self._sim = sim
        self._task = task
        # self.max_human_num = 6
        # self.human_num = task._human_num
        self.result_list = None  
        
        self.current_episode_id = None
        self.current_episode_scene_id = None
        self.current_episode_init_yaw = None
        
        self._top_down_map = None
        
        
        self.map_size = 101
        self.agent_diameter_pixels = 5
        self.agent_radius_pixels = self.agent_diameter_pixels // 2
        self.step_distance_pixels = 25/120
        self.angle_step_radians = 10/120 #20 / 120 #10 / 120
        
        # Define the center position
        self.center_position = (self.map_size // 2, self.map_size // 2)

        # Actions
        self.actions = ['left', 'right', 'forward']

        # Initialize agent direction (facing downwards)
        self.agent_angle = 3 * math.pi / 2  # 270 degrees
        
        self.future_step = kwargs['config']['future_step'] 
        # self.max_human_num = 6
        self.human_num = task._human_num
        self.episode = None

        self.trajectories = [(a1, a2, a3, a4, a5) for a1 in self.actions for a2 in self.actions for a3 in self.actions 
                    for a4 in self.actions for a5 in self.actions]
        
        super().__init__(*args, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return MPCWithHumanMapSensor.cls_uuid

    @staticmethod
    def _get_sensor_type(*args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3**self.future_step,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    @staticmethod
    def _initialize_map(self):
        """Initialize a zero map with the desired shape."""
        return np.zeros((101, 101, 3), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def move_agent(self , position, angle, action):
        precise_x, precise_y = position
        
        if action == 'left':
            angle += self.angle_step_radians
        elif action == 'right':
            angle -= self.angle_step_radians
        elif action == 'forward':
            # Calculate potential new position
            precise_x = precise_x + 25/120 * math.cos(angle)
            precise_y = precise_y - 25/120 * math.sin(angle)

        return (precise_x, precise_y ), angle
        

    def get_observation(self, task, observations, episode, *args: Any, **kwargs: Any):
        
        
        try:
            # Initialize agent direction (facing downwards)


            
            if self.current_episode_id != episode.episode_id or self.current_episode_scene_id != episode.scene_id:
                self.current_episode_id = episode.episode_id
                self.current_episode_scene_id = episode.scene_id
                # self.current_episode_init_yaw = camera_yaw
                self.episode = episode

            human_num = self._task._human_num
            # Initialize the map instead of the result list
            if human_num != self.human_num:
                self.human_num = human_num


            human_future_trajectory = task.measurements.measures.get("human_future_trajectory")._metric
            returnMap = self._initialize_map(self)
            # logger.info(f"check {precise_position[0]} {precise_position[1]} {heading_angle}")
            
            returnV = np.zeros((3**self.future_step,))
            for i, trajectory in enumerate(self.trajectories):
                robot_pos = np.array(self._sim.get_agent_state(0).position)[[0, 2]]
                precise_position = (float(robot_pos[0]), float(robot_pos[1]))


                T = self._sim.get_agent_data(0).articulated_agent.base_transformation
                forward = np.array([1.0, 0, 0])
                heading_angle = get_angle_to_pos(T.transform_vector(forward))
                
                for step, action in enumerate(trajectory):
                    px, py = precise_position
                    precise_position, heading_angle = self.move_agent(precise_position, heading_angle, action)
                    probot_pos = (px- robot_pos[0], py- robot_pos[1])
                    crobot_pos = (precise_position[0]- robot_pos[0], precise_position[1]- robot_pos[1])
                    
                    probot_pixel_x = round(50 + (probot_pos[0]) * 10) 
                    probot_pixel_y = round(50 + (probot_pos[1]) * 10)  
                    
                    crobot_pixel_x = round(50 + (crobot_pos[0]) * 10) 
                    crobot_pixel_y = round(50 + (crobot_pos[1]) * 10)  
                    
                    if (crobot_pixel_x, crobot_pixel_y) != (probot_pixel_x, probot_pixel_y):
                        cv2.line(returnMap, (probot_pixel_x, probot_pixel_y), (crobot_pixel_x, crobot_pixel_y), (255, 0, 0), 5)
                    # Draw the agent's position for this step
                    cv2.circle(returnMap, (crobot_pixel_x, crobot_pixel_y), 3, (255, 0, 0), -1)
            
            step_colors = [
                (255, 255, 255), # Cyan
                (255, 0, 255), # Magenta
                (0, 255, 255), # Yellow
                (128, 0, 128), # Purple
                (0, 128, 128)  # Teal
            ]
            if human_future_trajectory is not None:
                for key, trajectories in human_future_trajectory.items():
                    trajectories = np.array(trajectories)
                    trajectories = trajectories.astype('float32')
                    
                    for t in range(5):
                        human_position = trajectories[t, [0, 2]] - robot_pos
                        
                        # Calculate distance from robot to human
                        distance = np.linalg.norm(human_position)

                        # Only draw if the human is within 10 meters
                        if distance <= 10.0:
                            # Convert human position to pixel coordinates
                            human_pixel_x = round(50 + (human_position[0]) * 10)  # Centering the human position
                            human_pixel_y = round(50 + (human_position[1]) * 10)  # Centering the human position
                            
                            # Ensure pixel coordinates are within the map bounds
                            if 0 <= human_pixel_x < 101 and 0 <= human_pixel_y < 101 and t<self.future_step:
                                # Draw a filled circle for the human on the map
                                # future_map = np.ascontiguousarray(future_map)
                                cv2.circle(returnMap, (human_pixel_x, human_pixel_y), 3, step_colors[t], -1)
            return returnMap
        
        except Exception as e:
            logger.info("MPC with Human Sensor Exception", e)
            return np.zeros((101, 101, 5), dtype=np.float32)

cs = ConfigStore.instance()

cs.store(
    package="habitat.task.lab_sensors.oracle_shortest_path_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_shortest_path_sensor",
    node=OracleShortestPathSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_follower_sensor",
    group="habitat/task/lab_sensors",
    name="oracle_follower_sensor",
    node=OracleFollowerSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_velocity_sensor",
    group="habitat/task/lab_sensors",
    name="human_velocity_sensor",
    node=HumanVelocitySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.human_num_sensor",
    group="habitat/task/lab_sensors",
    name="human_num_sensor",
    node=HumanNumSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.risk_sensor",
    group="habitat/task/lab_sensors",
    name="risk_sensor",
    node=RiskSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.social_compass_sensor",
    group="habitat/task/lab_sensors",
    name="social_compass_sensor",
    node=SocialCompassSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_humanoid_future_trajectory",
    group="habitat/task/lab_sensors",
    name="oracle_humanoid_future_trajectory",
    node=OracleHumanoidFutureTrajectorySensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.oracle_humanoid_future_trajectory_map",
    group="habitat/task/lab_sensors",
    name="oracle_humanoid_future_trajectory_map",
    node=OracleHumanoidFutureTrajectoryMapSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.td_map",
    group="habitat/task/lab_sensors",
    name="td_map",
    node=TopDownMapSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.td_map_with_traj",
    group="habitat/task/lab_sensors",
    name="td_map_with_traj",
    node=TopDownMapWithTrajectorySensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.td_map_with_human",
    group="habitat/task/lab_sensors",
    name="td_map_with_human",
    node=TopDownMapWithHumanSensorConfig,
)

cs.store(
    package="habitat.task.lab_sensors.mpc_with_human",
    group="habitat/task/lab_sensors",
    name="mpc_with_human",
    node=MPCWithHumanSensorConfig,
)


cs.store(
    package="habitat.task.lab_sensors.mpc_with_human_map",
    group="habitat/task/lab_sensors",
    name="mpc_with_human_map",
    node=MPCWithHumanMapSensorConfig,
)


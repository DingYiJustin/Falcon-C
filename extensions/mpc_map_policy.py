# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple
import itertools

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from habitat.core.spaces import ActionSpace
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Policy, PolicyActionData
from habitat_baselines.utils.common import get_num_actions

import math
import copy

@baseline_registry.register_policy
class MapMPCPolicy(nn.Module, Policy):
    """
    :only apply one action, on my case, that is oracle random walk for humanoids.
    """

    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        orig_action_space: ActionSpace,
        num_envs: int,
        aux_loss_config,
        agent_name: Optional[str],
    ):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        self._num_envs: int = num_envs
        self._recurrent_hidden_size = (
            full_config.habitat_baselines.rl.ppo.hidden_size
        )
        self._device = None
        self.action_distribution_type = "categorical"
        self.turn_threshold = np.deg2rad(15.0)
        self.repeated_turn_count = [0] *  self._num_envs
        self.angle_diff = [0] *  self._num_envs
        self.max_repeated_turns = 3
        self.angle_threshold_for_forced_forward = np.deg2rad(5.0)
        
        self.actions = ['left', 'right', 'forward']
        self.trajectories = [(a1, a2, a3, a4, a5) for a1 in self.actions for a2 in self.actions for a3 in self.actions 
                    for a4 in self.actions for a5 in self.actions]
        
        # self.trajectories = [[]]
        nextTimeStep = []
        
        self.future_step =8 #5
        # for step, action in enumerate(trajectory):
        # for step in range(self.future_step): 
        #     while len(self.trajectories)>0:
        #         l = self.trajectories.pop()
        #         for action in self.actions:
        #             a = copy.deepcopy(l)
        #             a.append(action)
        #             nextTimeStep.append(a)
        #     self.trajectories = nextTimeStep
        #     nextTimeStep = []
        print("self.trajectories", self.trajectories)
        
        self.test = 0
                    

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        agent_name=None,
        **kwargs,
    ):
        if agent_name is None:
            if len(config.habitat.simulator.agents_order) > 1:
                raise ValueError(
                    "If there is more than an agent, you need to specify the agent name"
                )
            else:
                agent_name = config.habitat.simulator.agents_order[0]
        return cls(
            config=config.habitat_baselines.rl.policy[agent_name],
            full_config=config,
            observation_space=observation_space,
            action_space=action_space,
            orig_action_space=orig_action_space,
            num_envs=config.habitat_baselines.num_environments,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            agent_name=agent_name,
        )
    
    def to(self, device):
        self._device = device
        return super().to(device)
    
    @property
    def hidden_state_shape(self):
        return (
            self.num_recurrent_layers,
            self.recurrent_hidden_size,
        )

    @property
    def hidden_state_shape_lens(self):
        return [self.recurrent_hidden_size]

    @property
    def recurrent_hidden_size(self) -> int:
        return self._recurrent_hidden_size

    @property
    def num_recurrent_layers(self):
        return (0)

    @property
    def should_load_agent_state(self):
        return False
    
    @property
    def policy_action_space(self):
        """
        Fetches the policy action space for learning. If we are learning the HL
        policy, it will return its custom action space for learning.
        """
        return super().policy_action_space
    
    @property
    def policy_action_space_shape_lens(self):
        return [self._action_space]
    
    def parameters(self):
        return iter([nn.Parameter(torch.zeros((1,), device=self._device))])
    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        # We assign a value of 0. This is needed so that we can concatenate values in multiagent
        # policies
        return torch.zeros(rnn_hidden_states.shape[0], 1).to(
            rnn_hidden_states.device
        )
        
    def calculate_angle_to_target(self, agent_position, agent_yaw, target_position):
        delta_x = target_position[0] - agent_position[0]
        delta_y = target_position[2] - agent_position[2]
        
        target_angle = torch.atan2( -delta_y,delta_x)
        
        angle_diff = target_angle - agent_yaw
        return angle_diff

    def normalize_angle(self,angle):
        angle = (angle + torch.pi) % (2 * torch.pi) - torch.pi
        return angle

    def get_unique_actions(self, tensor):
        """
        Given a tensor with shape (batch_size, numOfTrajectories), return a 2D list with shape
        (batch_size, maximum 3) containing unique actions based on the max values in the tensor.

        :param tensor: A 2D torch tensor of shape (batch_size, numOfTrajectories)
        :return: A 2D list with shape (batch_size, maximum 3)
        """
        # print("tensor.shape", tensor.shape)
        batch_size, num_of_trajectories = tensor.shape
        result = []  # Initialize a list for each batch
        map = {'left':2, 'right':3, 'forward':1}
        # Process each row in the tensor
        for i in range(batch_size):
            # Find the maximum value in the current row
            max_value = tensor[i].max().item()  # Get the maximum value
            # Find indices of the maximum value
            max_indices = (tensor[i] == max_value).nonzero(as_tuple=True)[0]
            # print(i,'max_value', max_value, 'max_indices', max_indices)

            # Track unique actions for the current batch element
            unique_actions = set()
            for index in max_indices:
                # Get the first action for this trajectory
                first_action = self.trajectories[index.item()][0]  # Get the first action
                unique_actions.add(map[first_action])

                # If we already have 3 unique actions, we can stop
                if len(unique_actions) >= 3:
                    break

            # Convert the set to a list and append to the result, ensuring it has at most 3 elements
            # print('unique_actions', unique_actions)
            result.append(list(unique_actions))

        return result
    
    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
        **kwargs,
    ):
        # stop = 0
        # move_forward = 1
        # turn_left = 2
        # turn_right = 3
        batch_size = masks.shape[0]
        
        # print("observation action_to_follow", observations.keys())
        
        actions = torch.zeros(
            size=prev_actions.shape,
            device=masks.device,
            dtype=torch.int64,
        )
        for i in range(batch_size):
            if observations['pointgoal_with_gps_compass'][i, 0] <= 0.2:
                actions[i] = 0 
            else:
                current_point = observations['oracle_shortest_path_sensor'][i, 0, :]
                next_point = observations['oracle_shortest_path_sensor'][i, 1, :]

                self.angle_diff[i] = self.calculate_angle_to_target(current_point, 
                                observations['localization_sensor'][i, -1], next_point)
                if abs(self.angle_diff[i]) < self.turn_threshold or abs(self.angle_diff[i]) > 2 * torch.pi - self.turn_threshold:
                    actions[i] = 1  
                    self.repeated_turn_count[i] = 0 
                elif (self.angle_diff[i] > -2 * torch.pi + self.turn_threshold and self.angle_diff[i] < - torch.pi) or (self.angle_diff[i] > self.turn_threshold and self.angle_diff[i] < torch.pi):
                    if prev_actions[i] == 3:
                        self.repeated_turn_count[i] += 1
                    else:
                        self.repeated_turn_count[i] = 0
                    actions[i] = 2  # TURN_LEFT
                else:  
                    if prev_actions[i] == 2:
                        self.repeated_turn_count[i] += 1
                    else:
                        self.repeated_turn_count[i] = 0
                    actions[i] = 3  # TURN_RIGHT 

                if self.repeated_turn_count[i] >= self.max_repeated_turns:
                    actions[i] = 1  
                    self.repeated_turn_count[i] = 0  
        action_to_follow = actions
        # print("action_to_follow", action_to_follow)
        # print("observations['td_map_with_human']", torch.max( observations['td_map_with_human'], dim=1), torch.min( observations['td_map_with_human'], dim=1))
        # print("observations['mpc_with_human']", torch.max( observations['mpc_with_human'], dim=1), torch.min( observations['mpc_with_human'], dim=1))
        mpc_actions = self.get_unique_actions(observations['mpc_with_human'])
        print("observations['localization_sensor'][0, 2]", observations['localization_sensor'][0, 0], observations['localization_sensor'][0, 2], observations['localization_sensor'][0, -1])
        # if self.test < 5:
        #     actions = torch.ones(
        #         size=prev_actions.shape,
        #         device=masks.device,
        #         dtype=torch.int64,
        #     ) *2
        # else:
        #     actions = torch.ones(
        #         size=prev_actions.shape,
        #         device=masks.device,
        #         dtype=torch.int64,
        #     ) 
        # self.test +=1
        actions = torch.zeros(
            size=prev_actions.shape,
            device=masks.device,
            dtype=torch.int64,
        )
        # print("action.shape", actions.shape)
        
        for i in range(prev_actions.shape[0]):
            if action_to_follow[i].item() != 0:
                if len(mpc_actions[i]) < 3:
                    print('mpc_actions[i]', mpc_actions[i])
                if action_to_follow[i].item() not in mpc_actions[i]:
                    # print(len(mpc_actions), i)
                    # print(len(mpc_actions[i]))
                    actions[i] = mpc_actions[i][0] 
                    print('refine', mpc_actions[i])
                else:           
                    actions[i] =  action_to_follow[i]
        
        actions = actions.repeat(1, get_num_actions(self._action_space))
        
        use_action = actions

        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=rnn_hidden_states,
        )
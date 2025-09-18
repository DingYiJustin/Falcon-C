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

def apply_action(pos, action):
    x, y, heading = pos.clone()
    if action == 'forward':
        x = x + 25/120 * torch.cos(heading)
        y = y - 25/120 * torch.sin(heading)
        # print("aaa",25/120 * torch.cos(heading), 25/120 * torch.sin(heading))
    elif action == 'left':
        heading += 10/120
    elif action == 'right':
        heading -= 10/120
    return torch.tensor([x, y, heading])

def gen_future_positions(initial_pos, num_steps):
    batch_size = initial_pos.shape[0]
    actions = ['forward', 'left', 'right']
    actions_dict = {'forward':0, 'left':1, 'right':2}
    
    # Generate all possible action sequences
    action_sequences = list(itertools.product(actions, repeat=num_steps))
    num_sequences = len(action_sequences)
    
    # Prepare tensors to hold future positions and actions
    future_positions = torch.zeros((batch_size, num_sequences, num_steps, 3))
    future_actions = torch.zeros((batch_size, num_sequences, num_steps))
    
    for b in range(batch_size):
        pos = initial_pos[b]
        for seq_idx, action_seq in enumerate(action_sequences):
            current_pos = pos.clone()
            for step_idx, action in enumerate(action_seq):
                future_positions[b, seq_idx, step_idx] = current_pos
                future_actions[b, seq_idx, step_idx] = actions_dict[action]
                current_pos = apply_action(current_pos, action)
    
    return future_positions, future_actions

def generate_trajectory_mask(human_future_positions, robot_future_positions, distance_threshold=0.6):
    # Get shapes
    batch_size, num_humans, _, _ = human_future_positions.shape
    _, num_possible_pos, num_future_steps, num_robot_features = robot_future_positions.shape

    # Initialize the mask with False
    mask = torch.zeros((batch_size, num_future_steps), dtype=torch.float32)

    # Iterate over each future step
    for step in range(num_future_steps):
        # Get the robot future positions at this step
        robot_positions_at_step = robot_future_positions[:, :, step, :]  # Shape: (batch_size, num_possible_pos, 3)

        # Get the human future positions at this step
        human_positions_at_step = human_future_positions[:, :, step+1, :]  # Shape: (batch_size, num_humans, 2)

        # Calculate distances between robot positions and human positions
        for b in range(batch_size):
            for h in range(num_humans):
                # Get the position of the human
                human_pos = human_positions_at_step[b, h]  # Shape: (2,)
                for n in range(num_possible_pos):
                    # Calculate distance from each robot position to the human position
                    distances = torch.norm(robot_positions_at_step[b,n,:2] - human_pos)  # Shape: (num_robot,)

                    # Check if any distance is less than the threshold
                    if distances < distance_threshold:
                        mask[b, n, step] = 1.0 

    # Generate weights (1, 1/2, 1/4, 1/8, ...)
    weights = torch.tensor([1 / (2 ** i) for i in range(num_future_steps)], dtype=torch.float)  # Shape: (num_future_steps,)
    
    # Ensure weights are in the correct shape for broadcasting
    weights_tensor = weights.unsqueeze(0)  # Shape: (1, num_future_steps)

    # Perform weighted sum along the time dimension (dim=1)
    weighted_sum = torch.matmul(mask, weights_tensor.T)  # Shape: (batch_size, num_possible_pos)
    
    return weighted_sum.squeeze()  # Shape: (batch_size,num_possible_pos)


def obstacle_cost(self, state, actions, predictions):
        """
        Cost using 2D Gaussian around obstacles
        """
        # Distance to other agents
  #      rospy.loginfo("act: {} pred: {} state: {}\n\n".format(actions.shape, predictions.shape, state.shape))
        dx = actions[:, None, :, None, 0] - predictions[:, :, :, :, 0] #- (state[None, None, None, 1:, 4] + state[0, 4]) # N x S x T' x H
        dy = actions[:, None, :, None, 1] - predictions[:, :, :, :, 1] #- (state[None, None, None, 1:, 4] + state[0, 4]) # N x S x T' x H

                                                                                                                                                                                                            # rospy.loginfo(" dx:{} dy:{}".format(dx.shape, dy.shape))
        # Heading of "other agent"
        obs_theta = predictions[:, :, :, :, 3] # N x S x T' x H
        # Checking for static obstacles
        # static_obs = (np.linalg.norm(predictions[:, :, :, :, 2:4], axis=-1) < 0.01) # N x S x T' x H
        # Alpha calculates whether ego agent is in front or behind "other agent"
        alpha = self.wrap(np.arctan2(dy, dx) - obs_theta + np.pi/2.0) <= 0 # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo(" obs_theta:{} static_obs:{} alpha:{}".format(obs_theta.shape, static_obs.shape, alpha.shape))

        # Sigma values used to create 2D gaussian around obstacles for cost penalty
        sigma = np.where(alpha, self.sigma_r, self.sigma_h)
        sigma = static_obs + np.multiply(1-static_obs, sigma) # N x S x T' x H
        sigma_s = 1.0 * static_obs + self.sigma_s * (1 - static_obs) # N x S x T' x H
                                                                                                                                                                                                            # rospy.loginfo("s:{} ss:{}".format(sigma.shape, sigma_s.shape))

        # Variables used in cost_obs function based on sigma and obs_theta
        a = np.cos(obs_theta) ** 2 / (2 * sigma ** 2) + np.sin(obs_theta) ** 2 / (2 * sigma_s ** 2)
        b = np.sin(2 * obs_theta) / (4 * sigma ** 2) - np.sin(2 * obs_theta) / (4 * sigma_s ** 2)
        c = np.sin(obs_theta) ** 2 / (2 * sigma ** 2) + np.cos(obs_theta) ** 2 / (2 * sigma_s ** 2)

        cost = np.exp(-((a * dx ** 2) + (2 * b * dx * dy) +  (c * dy ** 2))) # N x S x T' x H
        cost = np.mean(cost, axis=3)
        cost = np.sum(cost, axis=-1)
        cost = -1 * cost
                                                                                                                                                                                                            # rospy.loginfo("c: {}\n\n".format(cost.shape))
        return self.Q_obs * (cost ** 2) # (N, S)

@baseline_registry.register_policy
class SimpleMPCPolicy(nn.Module, Policy):
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
        
        positions_gt_agent0 = observations["localization_sensor"][:, [0, 2, 3]]
        human_future_positions = observations['oracle_humanoid_future_trajectory'] + positions_gt_agent0[:,[0,1]].unsqueeze(dim=1).unsqueeze(dim=2)
        # action_to_follow = observations['action_to_follow']
        new_positions, future_actions = gen_future_positions(positions_gt_agent0, 1)
        new_positions = new_positions.to(device=human_future_positions.device)
        
        mask = generate_trajectory_mask(human_future_positions, new_positions)
        min_mask, mask = torch.min(mask, dim=1)
        first_actions = future_actions[torch.arange(mask.shape[0]), mask,0]
        mask = torch.ones((num_actions, batch_size), dtype=torch.float)

        # Iterate over the batch to set the correct indices to 0
        for b in range(batch_size):
            action_index = first_actions[b]  # Get the first action for this batch
            mask[action_index, b] = 0  # Set the mask at this action index to 0

        mask *= 10   
        
        action_mask = torch.ones((3, batch_size), dtype=torch.float, device=human_future_positions.device)

        # Iterate over the batch and update the mask
        
        for i in range(batch_size):
            action = action_to_follow[i].item()  # Get the action for the i-th instance
            
            # Set mask values based on the action label
            if action > 0:
                action_mask[action - 1, i] = 0
        action_mask *= 1
        
        final_mask = mask + action_mask
        min_values = torch.argmin(final_mask, dim=0).squeeze()+1
        action_to_follow = action_to_follow.reshape(batch_size)
        min_values_a = min_values
        min_values[action_to_follow == 0] = 0
        # print("action_to_follow", action_to_follow)
        # print("min_values", min_values)
        # Initialize empty action set based on the overall action space.
        actions = torch.zeros(
            (batch_size, get_num_actions(self._action_space)),
            device=masks.device,
        )
        actions = min_values.unsqueeze(dim=-1).repeat(1, get_num_actions(self._action_space))
        # actions[torch.arange(batch_size), min_values] = 1
        # print("action_to_follow", action_to_follow)
        # print("~action_to_follow", ~action_to_follow)
        # print("final_mask",final_mask)
        # print("min_values", min_values)
        # print("min_values_a", min_values_a)
        # print("actions", actions)
        # This will update the prev action
        use_action = actions

        return PolicyActionData(
            take_actions=actions,
            actions=use_action,
            rnn_hidden_states=rnn_hidden_states,
        )
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   DRL.py
@Time    :   2025/02/06 10:31:15
@Author  :   jzj
@Email   :   jzj.123@foxmail.com
@Version :   1.0
@description    :   xxxxx
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time

sys.path.append('../')

from network.model_test import *
from logger.logger import Logger
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from risk_perception.risk_perception import RiskPerception
import own_utils



def fuse_conv_and_bn(conv, bn):
        """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
        fusedconv = (
            torch.nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
            )
            .requires_grad_(False)
            .to(conv.weight.device)
        )

        # Prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

        # Prepare spatial bias
        b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


class SACAgent(object):

    def __init__(
        self,
        action_shape,
        device,
        logger:Logger,
        crop_action_shape=2,
        hidden_dim=128,
        discount=0.99,
        alpha_lr=1e-3,              # alpha
        alpha_beta=0.9,
        init_temperature=0.1,

        global_channels=[0, 1, 6, 7],# encoder
        local_channels=[2, 3, 4, 5],
        CNN_output_channels=64,
        state_dim=9,
        goal_dim=3,
        encoder_output_dim=64,
        goal_output_dim=32,
        state_output_dim=32,
        encoder_lr=1e-3,
        # encoder_out_dim=96,
        encoder_tau=0.005,        
        
        actor_lr=1e-3,              # actor
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,

        critic_lr=1e-3,             # critic
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,

        image_height=480,            # NeuFlow
        image_width=640,

        use_crop_action=False,

        guidance_weight=0.8,
        bc_steps=10000,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.action_shape = action_shape
        self.crop_action_shape = crop_action_shape
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.logger = logger
        self.use_crop_action = use_crop_action
        self.crop_size = (image_height // 8, image_width // 8)

        self.guidance_weight = guidance_weight
        self.bc_steps = bc_steps

        ########## Initial Encoder Network ##########
        self.encoder = featureEncoder(
            global_channels=global_channels, 
            local_channels=local_channels, 
            CNN_output_channels=CNN_output_channels,
            output_dim=encoder_output_dim,
            state_dim=state_dim,
            goal_dim=goal_dim,
            state_output_dim=state_output_dim,
            goal_output_dim=goal_output_dim,
        ).to(device)

        self.encoder_target = featureEncoder(
            global_channels=global_channels, 
            local_channels=local_channels, 
            CNN_output_channels=CNN_output_channels,
            state_dim=state_dim,
            goal_dim=goal_dim,
            output_dim=encoder_output_dim,
            state_output_dim=state_output_dim,
            goal_output_dim=goal_output_dim,
        ).to(device)

        self.encoder_target.load_state_dict(self.encoder.state_dict())

        ########## Initialize Actor Network ##########
        self.encoder_actor = featureEncoder(
            global_channels=global_channels, 
            local_channels=local_channels, 
            CNN_output_channels=CNN_output_channels,
            output_dim=encoder_output_dim,
            state_dim=state_dim,
            goal_dim=goal_dim,
            state_output_dim=state_output_dim,
            goal_output_dim=goal_output_dim,
        ).to(device)
        self.actor = Actor(
            32 + state_output_dim + goal_output_dim, hidden_dim, action_shape, crop_action_shape,
            actor_log_std_min, actor_log_std_max,
        ).to(device)             

        ########## Initialize Critic Network ##########
        self.critic = Critic(
            32 + state_output_dim + goal_output_dim, action_shape, crop_action_dim=crop_action_shape,
            hidden_dim=hidden_dim,
        ).to(device)

        self.critic_target = Critic(
            32 + state_output_dim + goal_output_dim, action_shape, crop_action_dim=crop_action_shape,
            hidden_dim=hidden_dim,
        ).to(device)
        
        self.critic_target.load_state_dict(self.critic.state_dict())

        ######### Initialize Auxiliary risk_perception module ##########
        self.risk_perception = RiskPerception(
            pixel_in_channels=CNN_output_channels,
            window_in_channels=CNN_output_channels,
            crop_size=self.crop_size,
            device=device
        )

        # autotune alpha
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        if use_crop_action:
            self.log_alpha_crop = torch.tensor(np.log(init_temperature)).to(device)
            self.log_alpha_crop.requires_grad = True
        # set target entropy to -|A|
        if self.use_crop_action:
            # self.target_entropy = -np.prod(action_shape + crop_action_shape)
            self.target_entropy = - torch.tensor([action_shape, crop_action_shape]).to(device)
        else:
            self.target_entropy = -np.prod(action_shape)

        # optimizers
        params = list(self.encoder_actor.parameters()) + list(self.actor.parameters())
        self.actor_optimizer = torch.optim.Adam(
            params, lr=actor_lr, betas=(actor_beta, 0.999)
        )

        params = list(self.critic.parameters()) + list(self.encoder.parameters())
        self.critic_optimizer = torch.optim.Adam(
            params, lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if use_crop_action:
            self.log_alpha_crop_optimizer = torch.optim.Adam(
                [self.log_alpha_crop], lr=alpha_lr, betas=(alpha_beta, 0.999)
            )

        auxiliary_params = list(self.encoder.parameters()) + list(self.risk_perception.parameters())
        self.auxiliary_optimizer = torch.optim.Adam(
            auxiliary_params, lr=encoder_lr
        )

        # NeuFlow: optical generator
        self.neuflow = NeuFlow().to(device)

        model_path = os.path.join(os.path.dirname(__file__), '../neuflow_model/neuflow_mixed.pth')
        checkpoint = torch.load(model_path, map_location='cuda')

        self.neuflow.load_state_dict(checkpoint['model'], strict=True)

        for m in self.neuflow.modules():
            if type(m) is ConvBlock:
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
                delattr(m, "norm1")  # remove batchnorm
                delattr(m, "norm2")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        
        self.neuflow.eval()
        self.neuflow.half()

        self.neuflow.init_bhwd(1, image_height, image_width, 'cuda')

        self.train()
        

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.encoder_actor.train(training)
        self.encoder_target.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)
        self.neuflow.train()


    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @property
    def alpha_crop(self):
        return self.log_alpha_crop.exp()
    
    def select_action(self, obs, goal, state):
        """
        输出最优动作

        Args:
            obs (Tensor): 当前观测值

        Returns:
            ndarray: 动作(均值)
        """
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            # goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # depth_downsample = obs[:, -1, :, :] * 10 # trans to meterDepth to compute risk
            # _, _, obs = self.encoder(obs, goal, state)
            _, _, obs = self.encoder_actor(obs, goal, state)
            # obs = obs.detach()
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False, use_crop_action=self.use_crop_action
            )
            
            if not self.use_crop_action:
                # use risk map to determine crop action
                crop_action = np.array([0, 0])
                # crop_action = self._get_crop_action_by_risk_map(depth_downsample)
                return mu.cpu().data.numpy(), crop_action
            else:
                mu_action, mu_crop = torch.split(mu, [self.action_shape, self.crop_action_shape], dim=1)
                return mu_action.cpu().data.numpy(), mu_crop.cpu().data.numpy()
        
    def sample_action(self, obs, goal, state):
        """
        输出随机采样动作

        Args:
            obs (Tensor): 当前观测值

        Returns:
            (ndarray, ndarray): 动作
        """
        with torch.no_grad():
            # obs = torch.FloatTensor(obs).to(self.device)
            # goal = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
            # state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # depth_downsample = obs[:, -1, :, :] * 10 # trans to meterDepth to compute risk
            # _, _, obs = self.encoder(obs, goal, state)
            _, _, obs = self.encoder_actor(obs, goal, state)
            # obs = obs.detach()
            mu, pi, log_pi, log_std = self.actor(obs, compute_log_pi=True, use_crop_action=self.use_crop_action)
            
            if not self.use_crop_action:
                # use risk map to determine crop action
                crop_action = np.array([0, 0])
                # crop_action = self._get_crop_action_by_risk_map(depth_downsample)
                return pi.cpu().data.numpy(), crop_action,  \
                log_pi.cpu().data.numpy(), None

            else:
                pi_action, pi_crop = torch.split(pi, [self.action_shape, self.crop_action_shape], dim=1)
                log_pi_action, log_pi_crop = torch.split(log_pi, [1, 1], dim=1)
                
                return pi_action.cpu().data.numpy(), pi_crop.cpu().data.numpy(), \
                log_pi_action.cpu().data.numpy(), log_pi_crop.cpu().data.numpy()

        
    def update_critic(self, obs, goal, state, action, crop_action, reward, next_obs, next_goal, next_state, not_done, step):
        """
        计算critic loss并更新

        Args:
            obs (Tensor): 观测值
            goal (Tensor): 目标
            state (Tensor): 状态
            action (Tensor): 动作
            reward (Tensor): 奖励
            next_obs (Tensor): 下一步的观测
            next_goal (Tensor): 下一步的目标
            next_state (Tensor): 下一步的状态
            not_done (Tensor): 是否是最后一步, 1表示不是
            step (int): 训练步数，用于记录数值
        """ 
        with torch.no_grad():
            _, _, next_obs = self.encoder_actor(next_obs, next_goal, next_state)
            # _, _, next_obs = self.encoder_target(next_obs, next_goal, next_state)
            _, policy_action, log_pi, _ = self.actor(next_obs, use_crop_action=self.use_crop_action)

            if not self.use_crop_action:
                target_Q1, target_Q2 = self.critic_target(next_obs, policy_action, self.use_crop_action)
                target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha.detach() * log_pi
                target_Q = reward + (not_done * self.discount * target_V)
            else:
                log_pi_action, log_pi_crop = torch.split(log_pi, [1, 1], dim=1)
                target_Q1, target_Q1_crop, target_Q2, target_Q2_crop = self.critic_target(next_obs, policy_action, self.use_crop_action)
                target_V = torch.min(target_Q1,
                                    target_Q2) - self.alpha.detach() * log_pi_action
                target_V_crop = torch.min(target_Q1_crop,
                                    target_Q2_crop) - self.alpha_crop.detach() * log_pi_crop
                target_Q = reward + (not_done * self.discount * target_V)
                target_Q_crop = reward + (not_done * self.discount * target_V_crop)
                self.logger.add_scalar('train_critic/target_Q_crop', target_Q_crop.mean().item(), step)
        _, _, obs = self.encoder(obs, goal, state)
        # get current Q estimates
        if not self.use_crop_action:
            current_Q1, current_Q2 = self.critic(obs, action, use_crop_action=self.use_crop_action)
        else:
            action = torch.concat([action, crop_action], dim=1)
            current_Q1, current_Q1_crop, current_Q2, current_Q2_crop = self.critic(obs, action, use_crop_action=self.use_crop_action)
            self.logger.add_scalar('train_critic/current_Q1_crop', current_Q1_crop.mean().item(), step)
            self.logger.add_scalar('train_critic/current_Q2_crop', current_Q2_crop.mean().item(), step)

        # critic loss
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if self.use_crop_action:
            critic_loss += F.mse_loss(current_Q1_crop, target_Q_crop) + F.mse_loss(current_Q2_crop, target_Q_crop)
        self.logger.add_scalar('train_critic/target_Q', target_Q.mean().item(), step)
        self.logger.add_scalar('train_critic/current_Q1', current_Q1.mean().item(), step)
        self.logger.add_scalar('train_critic/current_Q2', current_Q2.mean().item(), step)
        self.logger.add_scalar('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 40.)
        self.critic_optimizer.step()

        # self.critic.log(self.logger, step)  # TODO

    def update_actor_and_alpha(self, obs, goal, state, step):
        """
        更新actor和alpha

        Args:
            obs (_type_): 观测值
            goal (_type_): 目标
            state (_type_): 状态
            step (_type_): 训练步数，用于记录数值
        """
        # detach encoder, so we don't update it with the actor loss
        # _, _, obs = self.encoder(obs, goal, state)
        _, _, obs = self.encoder_actor(obs, goal, state)
        # obs = obs.detach()
        _, pi, log_pi, log_std = self.actor(obs, use_crop_action=self.use_crop_action)
        if not self.use_crop_action:
            actor_Q1, actor_Q2 = self.critic(obs, pi, use_crop_action=self.use_crop_action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi - actor_Q.detach()).mean()
        else:
            actor_Q1, actor_Q1_crop, actor_Q2, actor_Q2_crop = self.critic(obs, pi, use_crop_action=self.use_crop_action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_Q_crop = torch.min(actor_Q1_crop, actor_Q2_crop)
            
            log_pi_action, log_pi_crop = torch.split(log_pi, [1, 1], dim=1)

            actor_loss = (self.alpha.detach() * log_pi_action - actor_Q.detach()).mean() + (self.alpha_crop.detach() * log_pi_crop - actor_Q_crop.detach()).mean()

        self.logger.add_scalar('train_actor/loss', actor_loss, step)
        if self.use_crop_action:
            self.logger.add_scalar('train_actor/target_entropy', sum(self.target_entropy), step)
        else:
            self.logger.add_scalar('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        self.logger.add_scalar('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        self.actor_optimizer.step()

        # self.actor.log(self.logger, step) # TODO

        # optimize the alpha
        self.log_alpha_optimizer.zero_grad()
        if self.use_crop_action:
            self.log_alpha_crop_optimizer.zero_grad()
            alpha_crop_loss = (self.alpha_crop *
                                (-log_pi_crop - self.target_entropy[1]).detach()).mean()
            alpha_loss = (self.alpha *
                          (-log_pi - self.target_entropy[0]).detach()).mean()
            self.logger.add_scalar('train_alpha_crop/loss', alpha_crop_loss, step)
            self.logger.add_scalar('train_alpha_crop/value', self.alpha_crop, step)
            alpha_crop_loss.backward()
            self.log_alpha_crop_optimizer.step()
        else:
            alpha_loss = (self.alpha *
                        (-log_pi - self.target_entropy).detach()).mean()
            self.logger.add_scalar('train_alpha/loss', alpha_loss, step)
            self.logger.add_scalar('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_actor_and_alpha_with_expert(self, obs, goal, state, obs_expert, goal_expert, state_expert, action_expert, step, if_bc=False):
        """
        更新actor和alpha

        Args:
            obs (_type_): 观测值
            goal (_type_): 目标
            state (_type_): 状态
            step (_type_): 训练步数，用于记录数值
        """
        # detach encoder, so we don't update it with the actor loss
        # _, _, obs = self.encoder(obs, goal, state)
        _, _, obs = self.encoder_actor(obs, goal, state)
        # obs = obs.detach()
        _, pi, log_pi, log_std = self.actor(obs, use_crop_action=self.use_crop_action)
        if not self.use_crop_action:
            actor_Q1, actor_Q2 = self.critic(obs, pi, use_crop_action=self.use_crop_action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi - actor_Q.detach()).mean()
        else:
            actor_Q1, actor_Q1_crop, actor_Q2, actor_Q2_crop = self.critic(obs, pi, use_crop_action=self.use_crop_action)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_Q_crop = torch.min(actor_Q1_crop, actor_Q2_crop)
            
            log_pi_action, log_pi_crop = torch.split(log_pi, [1, 1], dim=1)

            actor_loss = (self.alpha.detach() * log_pi_action - actor_Q.detach()).mean() + (self.alpha_crop.detach() * log_pi_crop - actor_Q_crop.detach()).mean()

        guidence_loss = 0.0
        if if_bc:
            # _, _, obs_expert = self.encoder(obs_expert, goal_expert, state_expert)
            _, _, obs_expert = self.encoder_actor(obs_expert, goal_expert, state_expert)
            obs_expert = obs_expert.detach()
            _, pi_expert, _, _ = self.actor(obs_expert, use_crop_action=self.use_crop_action)
            if not self.use_crop_action:
                guidence_loss = self.guidance_weight * F.mse_loss(action_expert, pi_expert)
            else:
                guidence_loss =  self.guidance_weight * F.mse_loss(action_expert, pi_expert[:, :self.action_shape])

        self.logger.add_scalar('train_actor/guidence_loss', guidence_loss, step)
        
        actor_loss += guidence_loss
        self.logger.add_scalar('train_actor/loss', actor_loss, step)
        if self.use_crop_action:
            self.logger.add_scalar('train_actor/target_entropy', sum(self.target_entropy), step)
        else:
            self.logger.add_scalar('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        self.logger.add_scalar('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        self.actor_optimizer.step()

        # self.actor.log(self.logger, step) # TODO

        # optimize the alpha
        self.log_alpha_optimizer.zero_grad()
        if self.use_crop_action:
            self.log_alpha_crop_optimizer.zero_grad()
            alpha_crop_loss = (self.alpha_crop *
                                (-log_pi_crop - self.target_entropy[1]).detach()).mean()
            alpha_loss = (self.alpha *
                          (-log_pi - self.target_entropy[0]).detach()).mean()
            self.logger.add_scalar('train_alpha_crop/loss', alpha_crop_loss, step)
            self.logger.add_scalar('train_alpha_crop/value', self.alpha_crop, step)
            alpha_crop_loss.backward()
            self.log_alpha_crop_optimizer.step()
        else:
            alpha_loss = (self.alpha *
                        (-log_pi - self.target_entropy).detach()).mean()
            self.logger.add_scalar('train_alpha/loss', alpha_loss, step)
            self.logger.add_scalar('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    
    def update_encoder(self, obs, goal, state, crop_action, step):
        """
        使用辅助任务单独更新encoder

        Args:
            obs (_type_): _description_
            action (_type_): _description_
            step (_type_): _description_

        """
        # TODO
        depth_downsample = obs[:, -2, :, :] * 10 # trans to meterDepth to compute risk
        global_feature, local_feature, _ = self.encoder(obs, goal, state)

        labels = self.risk_perception.generate_labels(depth_map=depth_downsample, focus_coords=crop_action)

        outputs = self.risk_perception(global_feature, local_feature)

        window_target = labels.get('window_highrisk')
        losses = self.risk_perception.compute_loss(
            outputs,
            pixel_target=labels['pixel_risk'],
            window_target=window_target,
            lambda_pixel=1.0,
            lambda_window=1.0
        )

        total_loss = losses['total_loss']

        self.auxiliary_optimizer.zero_grad()

        self.logger.add_scalar('train_encoder/loss', total_loss, step)
        self.logger.add_scalar('train_encoder/loss_pixel', losses['pixel_loss'], step)
        self.logger.add_scalar('train_encoder/loss_window', losses['window_loss'], step)
        total_loss.backward()
        self.auxiliary_optimizer.step()
    
    def update(self, replay_buffer, step):
        """
        SAC算法更新各个网络

        Args:
            replay_buffer (ReplayBuffer): replay buffer
            step (int): 训练步数
        """
        torch.cuda.empty_cache()
        obs, goal, state, action, _, crop_action, _, reward, next_obs, next_goal, next_state, not_done = replay_buffer.sample()

        self.logger.add_scalar('train/batch_reward', reward.mean(), step)

        # update critic and encoder
        self.update_critic(obs, goal, state, action, crop_action, reward, next_obs, next_goal, next_state, not_done, step)
        
        # self.update_encoder(next_obs, next_goal, next_state, crop_action, step)

        # update actor and alpha
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, goal, state, step)

        # update target model
        if step % self.critic_target_update_freq == 0:
            own_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            own_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            own_utils.soft_update_params(
                self.encoder, self.encoder_target,
                self.encoder_tau
            )
    
    def update_with_expert_data(self, replay_buffer, pre_buffer, step):
        """
        使用专家数据，进行行为克隆，SAC算法更新各个网络

        Args:
            replay_buffer (ReplayBuffer): replay buffer
            pre_buffer (ReplayBuffer): expert replay buffer
            step (int): 训练步数
        """
        obs_agent, goal_agent, state_agent, action_agent, _, crop_action_agent, _, reward_agent, next_obs_agent, next_goal_agent, next_state_agent, not_done_agent = replay_buffer.sample()
        obs_expert, goal_expert, state_expert, action_expert, _, crop_action_expert, _, reward_expert, next_obs_expert, next_goal_expert, next_state_expert, not_done_expert = pre_buffer.sample()
        self.logger.add_scalar('train/batch_reward_agent', reward_agent.mean(), step)
        self.logger.add_scalar('train/batch_reward_expert', reward_expert.mean(), step)

        obs = torch.cat([obs_agent, obs_expert], dim=0)
        goal = torch.cat([goal_agent, goal_expert], dim=0)
        state = torch.cat([state_agent, state_expert], dim=0)
        action = torch.cat([action_agent, action_expert], dim=0)
        crop_action = torch.cat([crop_action_agent, crop_action_expert], dim=0)
        reward = torch.cat([reward_agent, reward_expert], dim=0)
        next_obs = torch.cat([next_obs_agent, next_obs_expert], dim=0)
        next_goal = torch.cat([next_goal_agent, next_goal_expert], dim=0)
        next_state = torch.cat([next_state_agent, next_state_expert], dim=0)
        not_done = torch.cat([not_done_agent, not_done_expert], dim=0)

        # update critic and encoder
        self.update_critic(obs, goal, state, action, crop_action, reward, next_obs, next_goal, next_state, not_done, step)
        
        # self.update_encoder(next_obs, next_goal, next_state, crop_action, step)

        # update actor and alpha
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha_with_expert(
                obs, goal, state, 
                obs_expert, goal_expert, state_expert, action_expert, step, if_bc=True)

        # update target model
        if step % self.critic_target_update_freq == 0:
            own_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            own_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            own_utils.soft_update_params(
                self.encoder, self.encoder_target,
                self.encoder_tau
            )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )  
        torch.save(
            self.encoder_actor.state_dict(), '%s/encoder_actor_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.encoder.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )
        self.encoder_actor.load_state_dict(
            torch.load('%s/encoder_actor_%s.pt' % (model_dir, step))
        )

    def _image_to_optical(self, img0, img1):
        """
        将输入的两帧图像转换为光流图

        Args:
            img0 (Tensor): 第一帧图像
            img1 (Tensor): 第二帧图像

        Returns:
            Tensor: 光流图
        """
        # img0 = img0.unsqueeze(0)  
        # img1 = img1.unsqueeze(0)
        with torch.no_grad():
            img0 = img0.half().to(self.device) # (1, C, H, W)
            img1 = img1.half().to(self.device)
            # start = time.time()
            flow = self.neuflow(img0, img1)[-1][0].detach()
            # end = time.time()
            # print("Optical Flow Time: ", end - start)
            flow = flow.unsqueeze(0)  # (2, H, W) -> (1, 2, H, W)
        
        return flow # (1, 2, H, W)

    def _perception(self, flow, depth, crop_midpoint, crop_size):
        '''
        Args:
            flow: torch.tensor [B, 2, H, W] 
            depth: torch.tensor [B, 2, H, W] :the continuous 2 frame depth
            crop_size: tuple/list (crop_width, crop_height)(H', W') :the size of the cropped area, which is also the size of the downsampled flow and depth
        Returns:
            perception_result: [B, 8, H', W'] 
            (8 channels where the first 2 channels are the downsample flow, 
            the next 2 channels are the cropped flow, 
            the next 2 channels are the coordinates of cropping area, 
            and the last 2 channels are the downsample depth)
        '''
        flow_downsample = torch.nn.functional.interpolate(flow, size=crop_size, mode='bilinear', align_corners=False)
        return flow_downsample
        depth_downsample = torch.nn.functional.interpolate(depth, size=crop_size, mode='bilinear', align_corners=False)

        batch, _, height, width = flow.shape

        crop_coords = torch.zeros((batch, 2, crop_size[0], crop_size[1]), device=flow.device)  # [B, 2, H', W']
        cropped_flow = torch.zeros((batch, 2, crop_size[0], crop_size[1]), device=flow.device)  # [B, 2, H', W']

        if crop_midpoint[0] - crop_size[0] // 2 < 0:
            crop_x1 = 0
            top_left_x = crop_size[0] // 2 - crop_midpoint[0]
            crop_size[0] = crop_midpoint[0] + crop_size[0] // 2
        elif crop_midpoint[0] + crop_size[0] // 2 > height - 1:
            crop_x1 = crop_midpoint[0] - crop_size[0] // 2
            top_left_x = 0
            crop_size[0] = height - crop_x1
        else:
            crop_x1 = crop_midpoint[0] - crop_size[0] // 2
            top_left_x = 0

        if crop_midpoint[1] - crop_size[1] // 2 < 0:
            crop_y1 = 0
            top_left_y = crop_size[1] // 2 - crop_midpoint[1]
            crop_size[1] = crop_midpoint[1] + crop_size[1] // 2
        elif crop_midpoint[1] + crop_size[1] // 2 > width - 1:
            crop_y1 = crop_midpoint[1] - crop_size[1] // 2
            top_left_y = 0
            crop_size[1] = width - crop_y1
        else:
            crop_y1 = crop_midpoint[1] - crop_size[1] // 2
            top_left_y = 0

        crop_x, crop_y = torch.meshgrid(torch.arange(crop_size[0], device=flow.device) + crop_x1, torch.arange(crop_size[1], device=flow.device) + crop_y1, indexing='ij')

        cropped_flow[:, :, top_left_x:top_left_x+crop_size[0], top_left_y:top_left_y+crop_size[1]]= flow[:, :, crop_x1:crop_x1+crop_size[0], crop_y1:crop_y1+crop_size[1]]

        crop_coords[:, : , top_left_x:top_left_x+crop_size[0], top_left_y:top_left_y+crop_size[1]]= torch.stack([crop_x, crop_y], dim=0).float().unsqueeze(0).repeat(batch, 1, 1, 1)  # [B, 2, H', W'] 
        crop_coords[:, 0, :, :] = crop_coords[:, 0, :, :] / (height - 1)  # normalize to [0, 1]
        crop_coords[:, 1, :, :] = crop_coords[:, 1, :, :] / (width - 1)
        
        perception_result = torch.cat([flow_downsample, cropped_flow, crop_coords, depth_downsample], dim=1)  # [B, 8, H', W']

        return perception_result
    

    def obs_to_input(self, observation, crop_size, crop_midpoint):
        '''
        Args:
            obs: list of [H, W, 8] : the original observation, which contains the RGB image and depth image of the last two frames
        Returns:
            input_tensor: torch.tensor [B, 8, H', W'] :the input for the actor and critic
        '''
        return torch.zeros([2, 60, 80]).unsqueeze(0)
        last_rgb_image = []
        rgb_image = []
        last_depth_image = []
        depth_image = []
        for obs in observation:
            last_rgb_image.append(torch.tensor(obs[:, :, 0:3], device=self.device))
            rgb_image.append(torch.tensor(obs[:, :, 3:6], device=self.device))
            last_depth_image.append(torch.tensor(obs[:, :, 6:7], device=self.device))
            depth_image.append(torch.tensor(obs[:, :, 7:8], device=self.device))
        last_rgb_image = torch.stack(last_rgb_image, dim=0).permute(0, 3, 1, 2)
        rgb_image = torch.stack(rgb_image, dim=0).permute(0, 3, 1, 2)
        last_depth_image = torch.stack(last_depth_image, dim=0).permute(0, 3, 1, 2)
        depth_image = torch.stack(depth_image, dim=0).permute(0, 3, 1, 2)

        flow = self._image_to_optical(last_rgb_image, rgb_image) # (1, 2, H, W)
        # normalize flow
        flow[:, 0, :, :] = (flow[:, 0, :, :] - torch.min(flow[:, 0, :, :])) / (torch.max(flow[:, 0, :, :]) - torch.min(flow[:, 0, :, :]))
        flow[:, 1, :, :] = (flow[:, 1, :, :] - torch.min(flow[:, 1, :, :])) / (torch.max(flow[:, 1, :, :]) - torch.min(flow[:, 1, :, :]))
        # normalize depth
        # last_depth_image = last_depth_image / 10.0
        # depth_image = depth_image / 10.0
        # depth = torch.cat([last_depth_image, depth_image], dim=1)
        # depth[depth == 0] = 1.0
        depth = None

        input_tensor = self._perception(flow, depth, crop_midpoint, crop_size)

        return input_tensor
    
    def _find_max_risk_area(self, risk_map, h, w):
        """
        Finds the h x w area with the maximum sum in a risk map.
        
        Args:
            risk_map (np.ndarray): A 2D array of shape (H, W) or (B, 1, H, W) with values 0-1. In general, B == 1.
            h (int): Height of the target area.
            w (int): Width of the target area.
            
        Returns:
            tuple: (max_sum, focus_row, focus_col) focus_row and focus_col are the coordinates of the mid point of the best window. 
        """
        risk_map = risk_map.cpu().numpy()
        if risk_map.ndim == 4:
            risk_map = risk_map.squeeze()

        H, W = risk_map.shape
        
        # Check if the requested window fits in the map
        if h > H or w > W:
            raise ValueError("Window size (h, w) is larger than the map (H, W)")

        # 1. Compute the Integral Image (Summed Area Table)
        # We pad with 0 to handle the top and left edges easily
        integral = np.zeros((H + 1, W + 1))
        
        # integral[y, x] = sum of rectangle from (0,0) to (y-1, x-1)
        integral[1:, 1:] = np.cumsum(np.cumsum(risk_map, axis=0), axis=1)

        max_sum = 0
        best_r, best_c = -1, -1

        # 2. Slide the window across the map
        # The bottom-right corner of the window can go up to (H, W)
        for r in range(h, H + 1):
            for c in range(w, W + 1):
                # Current window covers rows [r-h, r) and cols [c-w, c)
                
                # 3. Calculate sum in O(1) using the integral image
                # Sum = D - B - C + A
                # D: integral[r, c]
                # B: integral[r-h, c]
                # C: integral[r, c-w]
                # A: integral[r-h, c-w]
                
                current_sum = (integral[r, c] 
                            - integral[r - h, c] 
                            - integral[r, c - w] 
                            + integral[r - h, c - w])

                # 4. Update maximum
                if current_sum > max_sum:
                    max_sum = current_sum
                    best_r = r - h // 2
                    best_c = c - w // 2

        return best_r, best_c

    def _get_crop_action_by_risk_map(self, depth_map):
        labels = self.risk_perception.generate_labels(depth_map, None)
        risk_map = labels['pixel_risk']
        crop_size = (self.crop_size[0] // 8, self.crop_size[1] // 8)  # use smaller crop size for action selection
        focus_h, focus_w = self._find_max_risk_area(risk_map, crop_size[0], crop_size[1])
        crop_action = np.array([0, 0])
        # map focus coord from [0, H' - 1] into [-1, 1]
        crop_action[0] = 2 / (self.crop_size[0] - 1) * focus_h - 1
        crop_action[1] = 2 / (self.crop_size[1] - 1) * focus_w - 1

        return crop_action
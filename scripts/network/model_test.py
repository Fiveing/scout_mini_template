#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np



def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def tie_weights(src, trg):
    """
    将src和trg的神经网络层参数连接起来

    Args:
        src (_type_): nn.Module
        trg (_type_): nn.Module
    """
    assert type(src) == type(trg)
    if hasattr(src, 'weight') and hasattr(trg, 'weight'):
        trg.weight = src.weight
    if hasattr(src, 'bias') and hasattr(trg, 'bias'):
        trg.bias = src.bias


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros')

        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.norm1 = torch.nn.BatchNorm2d(out_planes)

        self.norm2 = torch.nn.BatchNorm2d(out_planes)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(x)

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))

        return x

    def forward_fuse(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x
    

class featureEncoder(torch.nn.Module):
    def __init__(self, 
                CNN_output_dim=64,
                hidden_dim=256,
                output_dim=64,
                state_dim=5,
                state_output_dim=32,
                crop_output_dim=16,
                pretrained_cnn=False,):
        super(featureEncoder, self).__init__()

        # self.global_channels = global_channels
        # self.local_channels = local_channels

        # self.test_global_cnn = ConvBlock(in_planes=2, out_planes=CNN_output_channels, kernel_size=3, stride=1, padding=1)

        # self.global_cnn = ConvBlock(in_planes=len(global_channels), out_planes=CNN_output_channels, kernel_size=3, stride=1, padding=1)
        # self.local_cnn = ConvBlock(in_planes=len(local_channels), out_planes=CNN_output_channels, kernel_size=3, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.global_local_fusion = nn.Sequential(
            nn.Linear(2 * CNN_output_dim + crop_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, state_output_dim),
            nn.ReLU(),
        )

        self.crop_action_mlp = nn.Sequential(
            nn.Linear(2, crop_output_dim),
            nn.ReLU(),
        )


        self.apply(weight_init)

        if pretrained_cnn:
            self.test_global_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.test_global_cnn = models.resnet18()

        self.local_cnn = models.resnet18()

        self.local_cnn.load_state_dict(self.test_global_cnn.state_dict())
            
        num_features = self.test_global_cnn.fc.in_features

        self.test_global_cnn.fc = nn.Linear(num_features, CNN_output_dim)

        self.test_global_cnn.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.local_cnn.fc.in_features

        self.local_cnn.fc = nn.Linear(num_features, CNN_output_dim)

        self.local_cnn.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, obs, state, last_crop_action=None):
        H, W = obs.shape[2], obs.shape[3]
        # global_feature = self.test_global_cnn(input) # [B, 2, H, W] - > [B, 64, H, W]
        # pooled_feature = self.avgpool(global_feature)
        # flat_feature = torch.flatten(pooled_feature, 1)
        state_feature = self.state_mlp(state)
        # laser_feature = self.laser_mlp(obs)
        # global_feature = self.test_global_cnn(obs) # [B, 2, H, W] - > [B, 64]
        global_obs = torch.cat([obs[:, 0:2, :, :], obs[:, -1:, :, :]], dim=1)
        global_obs = self.test_global_cnn.conv1(global_obs)
        global_obs = self.test_global_cnn.bn1(global_obs)
        global_obs = self.test_global_cnn.relu(global_obs)
        global_obs = self.test_global_cnn.maxpool(global_obs)
        global_obs = self.test_global_cnn.layer1(global_obs)
        global_obs = self.test_global_cnn.layer2(global_obs)
        global_obs = self.test_global_cnn.layer3(global_obs)
        global_obs = self.test_global_cnn.layer4(global_obs)
        global_feature = torch.nn.functional.interpolate(global_obs, (H, W), mode='bilinear', align_corners=False)
        global_obs = self.avgpool(global_obs)
        global_obs = torch.flatten(global_obs, 1)
        global_obs = self.test_global_cnn.fc(global_obs)
        
        local_obs = obs[:, 2:4, :, :]
        local_obs = self.local_cnn.conv1(local_obs)
        local_obs = self.local_cnn.bn1(local_obs)
        local_obs = self.local_cnn.relu(local_obs)
        local_obs = self.local_cnn.maxpool(local_obs)
        local_obs = self.local_cnn.layer1(local_obs)
        local_obs = self.local_cnn.layer2(local_obs)
        local_obs = self.local_cnn.layer3(local_obs)
        local_obs = self.local_cnn.layer4(local_obs)
        local_feature = torch.nn.functional.interpolate(local_obs, (H, W), mode='bilinear', align_corners=False)
        local_obs = self.avgpool(local_obs)
        local_obs = torch.flatten(local_obs, 1)
        local_obs = self.local_cnn.fc(local_obs)

        crop_action_feature = self.crop_action_mlp(last_crop_action)
        fusion_obs = torch.cat([global_obs, local_obs, crop_action_feature], dim=1)
        fusion_feature = self.global_local_fusion(fusion_obs)


        feature = torch.cat([fusion_feature, state_feature], dim=1)
        return global_feature, local_feature, feature

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        src_convs = [module for module in source.modules() if isinstance(module, nn.Conv2d)]
        trg_convs = [module for module in self.modules() if isinstance(module, nn.Conv2d)]

        assert len(src_convs) == len(trg_convs), "Number of convolutional layers does not match"

        for src, trg in zip(src_convs, trg_convs):
            # 假设 tie_weights 函数处理权重拷贝或参数绑定
            tie_weights(src=src, trg=trg)


class Actor(torch.nn.Module):
    def __init__(self, input_dim=64+32, hidden_dim=256, action_shape=2, crop_action_dim=2 ,log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape)
        )

        self.crop_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * crop_action_dim)
        )

        self.encoder = featureEncoder()

        self.apply(weight_init)


    def forward(self, obs, state, last_crop_action=None, compute_pi=True, compute_log_pi=True, use_crop_action=False):
        obs = self.encoder(obs, state, last_crop_action)[-1]
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            
        noise = torch.randn_like(mu)
        
        if compute_pi:
            std = log_std.exp()
            # sample
            pi = mu + noise * std
        else:
            pi = None

        if compute_log_pi:
            # sample
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        # apply squashing function
        mu, pi, log_pi = squash(mu, pi, log_pi)

        if use_crop_action:
            crop_mu, crop_log_std = self.crop_head(obs).chunk(2, dim=-1)
            
            crop_log_std = torch.tanh(crop_log_std)
            crop_log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (crop_log_std + 1)
            crop_noise = torch.randn_like(crop_mu)
            if compute_pi:
                # sample
                crop_pi = crop_mu + crop_noise * crop_log_std.exp()
            else:
                crop_pi = None
            if compute_log_pi:
                # sample
                crop_log_pi = gaussian_logprob(crop_noise, crop_log_std)
            else:
                crop_log_pi = None
            crop_mu, crop_pi, crop_log_pi = squash(crop_mu, crop_pi, crop_log_pi)
            mu = torch.concat([mu, crop_mu], dim=-1)
            if log_pi is not None:
                log_pi = torch.concat([log_pi, crop_log_pi], dim=-1)
            if pi is not None:
                pi = torch.concat([pi, crop_pi], dim=-1)
            log_std = torch.concat([log_std, crop_log_std], dim=-1)

        return mu, pi, log_pi, log_std
    

class QFunction (nn.Module):
    def __init__(
            self, feature_dim, action_dim, crop_action_dim=2,
            hidden_dim=128):
        super(QFunction, self).__init__()

        self.trunk = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.crop_trunk = nn.Sequential(
            nn.Linear(feature_dim + crop_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(weight_init)

    def forward(self, obs_feature, action, use_crop_action=False):
        # check if the batch sizes of obs and action are consistent
        assert obs_feature.size(0) == action.size(0)

        if use_crop_action:
            inputs = torch.cat([obs_feature, action[:, :-2]], dim=1)
            crop_inputs = torch.cat([obs_feature, action[:, -2:]], dim=1)
            q, q_crop = self.trunk(inputs), self.crop_trunk(crop_inputs)

            return q, q_crop
        else:
            inputs = torch.cat([obs_feature, action], dim=1)
            q = self.trunk(inputs)

            return q


class Critic(nn.Module):
    def  __init__(self,
                  feature_dim, action_dim, crop_action_dim=2, hidden_dim=128):
        super(Critic, self).__init__()

        self.encoder = featureEncoder(pretrained_cnn=False)

        self.Q1 = QFunction(feature_dim, action_dim, crop_action_dim=crop_action_dim, hidden_dim=hidden_dim)
        self.Q2 = QFunction(feature_dim, action_dim, crop_action_dim=crop_action_dim, hidden_dim=hidden_dim)

        self.apply(weight_init)

    def forward(self, obs, state, action, last_crop_action, use_crop_action=False):

        if not use_crop_action:
            _, _, obs = self.encoder(obs, state, last_crop_action)
            q1 = self.Q1(obs, action, use_crop_action)
            q2 = self.Q2(obs, action, use_crop_action)

            return q1, q2
        else:
            _, _, obs = self.encoder(obs, state, last_crop_action)
            q1, q1_crop = self.Q1(obs, action, use_crop_action)
            q2, q2_crop = self.Q2(obs, action, use_crop_action)

            return q1, q1_crop, q2, q2_crop




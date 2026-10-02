import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network.model_test import Critic, Actor
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from risk_perception.risk_perception import RiskPerception
import numpy as np


# --- 辅助函数：定义一个通用的 MLP ---
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

def create_mlp(input_dim, output_dim, hidden_dims, activation=nn.ReLU):
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


# --- 辅助类：定义策略网络 (Actor) ---
class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims):
        super().__init__()
        self.net = create_mlp(obs_dim, hidden_dims[-1], hidden_dims[:-1])
        self.mu_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.act_dim = act_dim
        # 用于将动作缩放到环境范围 [min_action, max_action]
        # 假设环境动作范围是 [-2, 2] m/s
        self.action_scale = 2.0
        self.action_bias = 0.0

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        根据观测值计算动作和对数概率。
        """
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, -20, 2) # 限制 log_std 范围

        std = torch.exp(log_std)
        pi_distribution = torch.distributions.Normal(mu, std)

        if deterministic:
            # 确定性模式，使用均值作为动作
            pi_action = mu
        else:
            # 随机模式，从分布中采样动作
            pi_action = pi_distribution.rsample() 

        if with_logprob:
            # 计算 Log-likelihood，并应用重参数化技巧和 Squashing 校正
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # 应用 Squashing 函数 (tanh)
        pi_action = torch.tanh(pi_action)
        # 缩放动作到环境范围
        pi_action = self.action_scale * pi_action + self.action_bias

        return pi_action, logp_pi
    
    
    
    # --- 辅助类：简单的回放缓冲区 ---
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, state_dim, act_dim, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # 初始化存储张量
        self.obs_buf = torch.zeros((capacity, *obs_dim), dtype=torch.float32, device='cpu')
        self.next_obs_buf = torch.zeros((capacity, *obs_dim), dtype=torch.float32, device='cpu')
        
        self.state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, device='cpu')
        self.next_state_buf = torch.zeros((capacity, state_dim), dtype=torch.float32, device='cpu')
        
        self.act_buf = torch.zeros((capacity, act_dim), dtype=torch.float32, device='cpu')
        self.crop_act_buf = torch.zeros((capacity, 2), dtype=torch.float32, device='cpu')
        self.last_crop_act_buf = torch.zeros((capacity, 2), dtype=torch.float32, device='cpu')
        self.rew_buf = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')
        self.done_buf = torch.zeros((capacity, 1), dtype=torch.float32, device='cpu')

    def add(self, obs, state, action, reward, next_obs, next_state, done, crop_action=None, last_crop_action=None):
        # 假设输入张量形状为 (N, D)，N 为 num_envs
        N = obs.shape[0]
        
        # 确保数据在 CPU 上存储，以节省 GPU 内存
        obs = obs.cpu()
        state = state.cpu()
        action = action.cpu()
        reward = reward.cpu()
        next_obs = next_obs.cpu()
        next_state = next_state.cpu()
        done = done.cpu().float()
        if crop_action is not None:
            crop_action = crop_action.cpu()
        if last_crop_action is not None:
            last_crop_action = last_crop_action.cpu()

        # 计算存储位置
        end_ptr = self.ptr + N
        if end_ptr > self.capacity:
            # 处理循环存储
            overlap = end_ptr - self.capacity
            
            # 存储尾部
            # self._add_slice(self.ptr, self.capacity, N - overlap, obs, action, reward, next_obs, done)
            self._add_slice(self.ptr, self.capacity, 0, obs, state, action, reward, next_obs, next_state, done, crop_action, last_crop_action)
            
            # 存储头部
            # self._add_slice(0, overlap, 0, obs, action, reward, next_obs, done)
            self._add_slice(0, overlap, N - overlap, obs, state, action, reward, next_obs, next_state, done, crop_action, last_crop_action)
            self.ptr = overlap
        else:
            self._add_slice(self.ptr, end_ptr, 0, obs, state, action, reward, next_obs, next_state, done, crop_action, last_crop_action)
            self.ptr = end_ptr % self.capacity

        self.size = min(self.size + N, self.capacity)

    def _add_slice(self, start_idx, end_idx, slice_start, obs, state, act, rew, next_obs, next_state, done, crop_action=None, last_crop_action=None):
        count = end_idx - start_idx
        self.obs_buf[start_idx:end_idx] = obs[slice_start : slice_start + count]
        self.state_buf[start_idx:end_idx] = state[slice_start : slice_start + count]
        self.act_buf[start_idx:end_idx] = act[slice_start : slice_start + count]
        self.rew_buf[start_idx:end_idx] = rew[slice_start : slice_start + count]
        self.next_obs_buf[start_idx:end_idx] = next_obs[slice_start : slice_start + count]
        self.next_state_buf[start_idx:end_idx] = next_state[slice_start : slice_start + count]
        self.done_buf[start_idx:end_idx] = done[slice_start : slice_start + count]

        if crop_action is not None:
            self.crop_act_buf[start_idx:end_idx] = crop_action[slice_start : slice_start + count]
        if last_crop_action is not None:
            self.last_crop_act_buf[start_idx:end_idx] = last_crop_action[slice_start : slice_start + count]
            
    def sample(self, batch_size):
        # 随机采样索引
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # 将采样数据移回 device
        data = dict(obs=self.obs_buf[idxs].to(self.device),
                    state=self.state_buf[idxs].to(self.device),
                    act=self.act_buf[idxs].to(self.device),
                    crop_act=self.crop_act_buf[idxs].to(self.device),
                    rew=self.rew_buf[idxs].to(self.device),
                    next_obs=self.next_obs_buf[idxs].to(self.device),
                    next_state=self.next_state_buf[idxs].to(self.device),
                    done=self.done_buf[idxs].to(self.device),
                    last_crop_act=self.last_crop_act_buf[idxs].to(self.device))
        return data
    
    
    
    
    # --- 核心类：SAC Agent ---
class SACAgent:
    def __init__(self, obs_dim, action_dim, cfg, logger):
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.device = cfg["device"]
        self.batch_size = cfg["batch_size"]
        self.action_dim = action_dim
        self.crop_size = cfg["crop_size"]
        self.use_crop_action = cfg.get("use_crop_action", False)
        self.logger = logger
        if not self.use_crop_action:
            self.target_entropy = - action_dim # 目标熵值，用于自动调节 alpha
        else:
            self.target_entropy = [- action_dim, -2.0] # 分别为动作熵和裁剪动作熵的目标值

        # --- 策略网络 (Actor) ---
        # self.actor = SquashedGaussianActor(obs_dim, action_dim, cfg["actor_hidden_dims"]).to(self.device)
        # log_std_max 决定探索噪声 sigma 的上限. 原来是 2 (sigma<=7.4), 对"动作=速度"的小车来说太大:
        # 采样动作被噪声淹没 -> critic 看不到角速度的作用 -> 学不会转向. 这里限制到 sigma<=0.3
        self.actor: Actor = Actor(
            64 + 32,
            action_shape=action_dim,
            log_std_min=cfg.get("log_std_min", -20),
            log_std_max=cfg.get("log_std_max", 2),
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        
        # --- 价值网络 (Critic) ---
        # q_input_dim = obs_dim + action_dim
        
        # Critic Q1 和 Q2
        self.critic = Critic(64 + 32, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        self.critic_target = Critic(64 + 32, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # --- 温度参数 (Alpha) ---
        # 注意: 原来的 log_alpha=0 (alpha=1) 对"动作直接决定速度"的小车来说探索噪声过大:
        # 策略输出 mu 还没长大, 采样的动作就已经被 sigma 淹没, critic 根本看不出角速度的作用,
        # 于是永远学不会"朝目标转向". 这里允许把初始温度调小.
        init_log_alpha = cfg.get("init_log_alpha", 0.0)
        self.log_alpha_max = cfg.get("log_alpha_max", None)
        if not self.use_crop_action:
            self.log_alpha = torch.full((1,), init_log_alpha, requires_grad=True, device=self.device)
        else:
            self.log_alpha = torch.full((2,), init_log_alpha, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=cfg["alpha_learning_rate"])
        self.risk_perception = RiskPerception(
            pixel_in_channels=512,
            window_in_channels=512,
            crop_size=self.crop_size,
            device=self.device
        ) # 后续添加窗口级高危判别器
        # self.alpha = self.log_alpha.exp()

        self.auxiliary_optimizer = optim.Adam(list(self.risk_perception.parameters()) + list(self.critic.encoder.parameters()), lr=cfg["learning_rate"])
        
        # --- 回放缓冲区 ---
        self.buffer = ReplayBuffer(cfg["buffer_size"], obs_dim, cfg["state_dim"], action_dim, self.device)
        
        # 学习计数器，用于交替更新
        self.update_count = 0

        # NeuFlow: optical generator
        self.neuflow = NeuFlow().to(self.device)

        model_path = os.path.join(os.path.dirname(__file__), '../neuflow_model/neuflow_mixed.pth')
        checkpoint = torch.load(model_path, map_location=self.device)

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

        self.neuflow.init_bhwd(1, 480, 640, 'cuda')
    
    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, state, last_crop_action=None, deterministic=False):
        """策略的动作输出接口，用于环境交互。"""
        with torch.no_grad():
            if not deterministic:
                _, a, logp_a, _ = self.actor(obs, state, last_crop_action=last_crop_action, use_crop_action=self.use_crop_action)
            else:
                a, _, logp_a, _ = self.actor(obs, state, last_crop_action=last_crop_action, compute_pi=False, compute_log_pi=False, use_crop_action=self.use_crop_action)
            if self.use_crop_action:
                a, a_crop = torch.split(a, [self.action_dim, 2], dim=-1)
                if logp_a is not None:
                    logp_a, logp_a_crop = torch.split(logp_a, [1, 1], dim=-1)
                else:
                    logp_a_crop = None
            else:
                # depth = obs[:, -1:, :, :]
                a_crop = None #self._get_crop_action_by_risk_map(depth_map=depth)
                logp_a_crop = None
            # a, logp_a = self.actor(obs, deterministic=deterministic, with_logprob=True)
            return a, a_crop, logp_a, logp_a_crop


    def update_critic(self, obs, state, next_obs, next_state, act, rew, done, step, crop_act=None, last_crop_act=None):
        """更新 Critic 网络。"""
        with torch.no_grad():
            # 使用目标网络计算 V(s')
            _, next_a, next_logp_a, _ = self.actor(next_obs, next_state, last_crop_action=crop_act, use_crop_action=self.use_crop_action)
            if self.use_crop_action:
                next_logp_a, next_logp_a_crop = torch.split(next_logp_a, [1, 1], dim=-1)
                q1_target, q1_target_crop, q2_target, q2_target_crop = self.critic_target(next_obs, next_state, next_a, last_crop_action=crop_act, use_crop_action=True)
                q_target = torch.min(q1_target, q2_target)
                q_crop_target = torch.min(q1_target_crop, q2_target_crop)
                v_target = q_target - self.alpha[0].detach() * next_logp_a.view(-1, 1)
                v_crop_target = q_crop_target - self.alpha[1].detach() * next_logp_a_crop.view(-1, 1)
                y_target = rew + self.gamma * (1 - done) * v_target
                y_crop_target = rew + self.gamma * (1 - done) * v_crop_target
            else:
                q1_target, q2_target = self.critic_target(next_obs, next_state, next_a, last_crop_action=crop_act, use_crop_action=False)
                q_target = torch.min(q1_target, q2_target)
                
                # V(s') = Q_target - alpha * log_pi(a'|s')
                v_target = q_target - self.alpha.detach() * next_logp_a.view(-1, 1)
                
                # Bellman 目标: y = r + gamma * (1 - done) * V(s')
                y_target = rew + self.gamma * (1 - done) * v_target

        # 计算 Q1 和 Q2 的损失
        # q_input_current = torch.cat([obs, act], dim=-1)
        # q1 = self.critic1(q_input_current)
        # q2 = self.critic2(q_input_current)
        if not self.use_crop_action:
            q1, q2 = self.critic(obs, state, act, last_crop_action=last_crop_act, use_crop_action=False)
        else:
            act = torch.cat([act, crop_act], dim=-1)
            q1, q1_crop, q2, q2_crop = self.critic(obs, state, act, last_crop_action=last_crop_act, use_crop_action=True)
            self.logger.add_scalar('train_critic/current_Q1_crop', q1_crop.mean().item(), step)
            self.logger.add_scalar('train_critic/current_Q2_crop', q2_crop.mean().item(), step)
        
        q1_loss = F.mse_loss(q1, y_target)
        q2_loss = F.mse_loss(q2, y_target)
        q_loss = q1_loss + q2_loss
        if self.use_crop_action:
            q1_crop_loss = F.mse_loss(q1_crop, y_crop_target)
            q2_crop_loss = F.mse_loss(q2_crop, y_crop_target)
            q_crop_loss = q1_crop_loss + q2_crop_loss
            q_loss += q_crop_loss
        # 梯度裁减
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 40.)

        self.logger.add_scalar('train_critic/target_Q', y_target.mean().item(), step)
        self.logger.add_scalar('train_critic/current_Q1', q1.mean().item(), step)
        self.logger.add_scalar('train_critic/current_Q2', q2.mean().item(), step)
        self.logger.add_scalar('train_critic/loss', q_loss, step)
        # 优化 Q 网络
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, state, step, last_crop_action=None):
        """更新 Actor 和 Alpha。"""
        # 冻结 Q 网络参数，以计算 Actor 损失
        for p in self.critic.parameters():
            p.requires_grad = False
            
        # 计算新策略下的 Q 值和熵
        _, pi_action, logp_pi, _ = self.actor(obs, state, last_crop_action=last_crop_action, use_crop_action=self.use_crop_action)
        # q_input_pi = torch.cat([obs, pi_action], dim=-1)
        # q1_pi = self.critic1(q_input_pi)
        # q2_pi = self.critic2(q_input_pi)
        if not self.use_crop_action:
            q1_pi, q2_pi = self.critic(obs, state, pi_action, last_crop_action=last_crop_action)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * logp_pi.unsqueeze(-1) - q_pi).mean()
        else:
            q1_pi, q1_pi_crop, q2_pi, q2_pi_crop = self.critic(obs, state, pi_action, last_crop_action=last_crop_action, use_crop_action=True)
            q_pi = torch.min(q1_pi, q2_pi)
            q_pi_crop = torch.min(q1_pi_crop, q2_pi_crop)

            logp_pi, logp_pi_crop = torch.split(logp_pi, [1, 1], dim=-1)
            actor_loss = (self.alpha.detach() * logp_pi.unsqueeze(-1) - q_pi).mean() + (self.alpha.detach() * logp_pi_crop.unsqueeze(-1) - q_pi_crop).mean()
        # 梯度裁减
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        # Actor 损失：L_pi = E [alpha * log_pi(a|s) - Q_pi]
        
        self.logger.add_scalar('train_actor/loss', actor_loss, step)

        # 优化 Actor 网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 解冻 Q 网络参数
        for p in self.critic.parameters():
            p.requires_grad = True

        # ---------------------------
        # 步骤 3: 更新温度参数 (Alpha)
        # ---------------------------
        
        # Alpha 损失：L_alpha = E [-alpha * (log_pi(a|s) + H_target)]
        if not self.use_crop_action:
            alpha_loss = (self.alpha * (-logp_pi.detach() - self.target_entropy)).mean()
            self.logger.add_scalar('train_alpha/loss', alpha_loss, step)
            self.logger.add_scalar('train_alpha/value', self.alpha, step)
        else:
            alpha_crop_loss = (self.alpha[1] * (-logp_pi_crop.detach() - self.target_entropy[1])).mean()
            alpha_loss = (self.alpha[0] * (-logp_pi.detach() - self.target_entropy[0])).mean() + alpha_crop_loss
            self.logger.add_scalar('train_alpha/loss', alpha_loss, step)
            self.logger.add_scalar('train_alpha/value', self.alpha[0], step)
            self.logger.add_scalar('train_alpha_crop/value', self.alpha[1], step)
            self.logger.add_scalar('train_alpha_crop/loss', alpha_crop_loss, step)
            alpha_loss += alpha_crop_loss
        # 优化 Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # alpha 上限: 实测 alpha 会一路涨到 0.6+, 而 actor 梯度里的熵噪声 ∝ alpha/sigma,
        # 涨上去之后策略均值被噪声带偏, 成功率从峰值 0.21 掉到 0.02~0.08.
        # 加个上限把探索权重钉在"表现最好"的区间.
        if getattr(self, "log_alpha_max", None) is not None:
            self.log_alpha.data.clamp_(max=self.log_alpha_max)
    
    def update_encoder(self, obs, state, crop_action, step, last_crop_action=None):
        depth_down_sample = obs[:, -1:, :, :]
        global_feature, local_feature, _ = self.critic.encoder(obs, state, last_crop_action=last_crop_action)

        labels = self.risk_perception.generate_labels(depth_down_sample, crop_action)

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
    
    def update(self, step):
        """执行一个 SAC 学习步骤。"""
        torch.cuda.empty_cache()
        data = self.buffer.sample(self.batch_size)
        
        obs, state, next_obs, next_state, act, rew, done = data['obs'], data['state'], data['next_obs'], data['next_state'], data['act'], data['rew'], data['done']
        crop_action = data['crop_act']
        last_crop_action = data['last_crop_act']

        self.update_count += 1

        # ---------------------------
        # 步骤 1: 更新 Q 网络 (Critic)
        # ---------------------------
        self.update_critic(obs, state, next_obs, next_state, act, rew, done, step, crop_act=crop_action, last_crop_act=last_crop_action)
        # self.update_encoder(obs, state, crop_action, step, last_crop_action=last_crop_action)
        # ---------------------------
        # 步骤 2: 更新策略网络 (Actor)
        # ---------------------------
        self.update_actor_and_alpha(obs, state, step, last_crop_action=last_crop_action)

        # ---------------------------
        # 步骤 4: 软更新目标 Q 网络
        # ---------------------------
        self._polyak_update(self.critic, self.critic_target)

    def _polyak_update(self, net, target_net):
        """软更新目标网络参数: theta_targ = tau * theta + (1 - tau) * theta_targ"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
            """保存模型、优化器以及 alpha 参数到指定路径"""
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict' : self.critic.state_dict(),
                'critic_target_state_dict' : self.critic_target.state_dict(),
                'risk_perception_state_dict': self.risk_perception.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
                'log_alpha': self.log_alpha,
                'update_count': self.update_count
            }, path)
            print(f"模型已保存至: {path}")

    def load(self, path, use_log_alpha=True):
        """从指定路径加载所有参数"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.risk_perception.load_state_dict(checkpoint['risk_perception_state_dict'])
        if use_log_alpha:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            
            self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
        self.update_count = checkpoint['update_count']
        print(f"模型已从 {path} 加载成功")
    
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
        depth_downsample = torch.nn.functional.interpolate(depth, size=crop_size, mode='bilinear', align_corners=False)

        batch, _, height, width = flow.shape

        crop_coords = torch.zeros((batch, 2, crop_size[0], crop_size[1]), device=flow.device)  # [B, 2, H', W']
        cropped_flow = torch.zeros((batch, 2, crop_size[0], crop_size[1]), device=flow.device)  # [B, 2, H', W']
        
        if isinstance(crop_midpoint[0], torch.Tensor):
            crop_midpoint = (crop_midpoint[0].cpu(), crop_midpoint[1].cpu()) 
        else:
            crop_midpoint = (torch.tensor(crop_midpoint[0]), torch.tensor(crop_midpoint[1])) 
        crop_x1 = crop_midpoint[0] - crop_size[0] // 2
        crop_x1 = crop_x1.clamp(0, height - 1)
        crop_x2 = crop_midpoint[0] + crop_size[0] // 2
        crop_x2 = crop_x2.clamp(0, height - 1)
        crop_y1 = crop_midpoint[1] - crop_size[1] // 2
        crop_y1 = crop_y1.clamp(0, width - 1)
        crop_y2 = crop_midpoint[1] + crop_size[1] // 2
        crop_y2 = crop_y2.clamp(0, width - 1)
        top_left_x = crop_size[0] // 2 - (crop_midpoint[0] - crop_x1)
        top_left_y = crop_size[1] // 2 - (crop_midpoint[1] - crop_y1)

        down_right_x = top_left_x + (crop_x2 - crop_x1)
        down_right_y = top_left_y + (crop_y2 - crop_y1)

        seq_x = torch.arange(crop_size[0]).unsqueeze(0)
        seq_y = torch.arange(crop_size[1]).unsqueeze(0)

        top_left_x = top_left_x.reshape(-1, 1)
        top_left_y = top_left_y.reshape(-1, 1)
        down_right_x = down_right_x.reshape(-1, 1)
        down_right_y = down_right_y.reshape(-1, 1)

        index_x = ((seq_x >= top_left_x) & (seq_x < down_right_x)).reshape(-1, crop_size[0], 1)  # [B, H', 1]
        index_y = ((seq_y >= top_left_y) & (seq_y < down_right_y)).reshape(-1, 1, crop_size[1])  # [B, 1, W']

        index_map = index_x & index_y # [B, H', W']
        index_map = index_map.unsqueeze(1).expand_as(crop_coords) # [B, 2, H', W']

        seq_x = torch.arange(height).unsqueeze(0)
        seq_y = torch.arange(width).unsqueeze(0)

        crop_x1 = crop_x1.reshape(-1, 1)
        crop_y1 = crop_y1.reshape(-1, 1)
        crop_x2 = crop_x2.reshape(-1, 1)
        crop_y2 = crop_y2.reshape(-1, 1)

        crop_index_x = ((seq_x >= crop_x1) & (seq_x < crop_x2)).reshape(-1, height, 1)  # [B, H, 1]
        crop_index_y = ((seq_y >= crop_y1) & (seq_y < crop_y2)).reshape(-1, 1, width)  # [B, 1, W]

        crop_index_map = crop_index_x & crop_index_y # [B, H, W]
        crop_index_map = crop_index_map.unsqueeze(1).expand_as(flow) # [B, 2, H, W]

        cropped_flow[index_map] = flow[crop_index_map]

        coords_x, coords_y = torch.meshgrid(seq_x.squeeze(), seq_y.squeeze(), indexing='ij')

        coords = torch.stack([coords_x, coords_y], dim=0).float()  # [2, H, W]
        coords = coords.unsqueeze(0).expand(batch, -1, -1, -1)  # [B, 2, H, W]
        
        crop_coords[index_map] = coords[crop_index_map].to(flow.device)

        # crop_x = torch.arange(crop_size[0], device=flow.device) + crop_x1.reshape(-1, 1).to(device=flow.device)  # [B, H']
        # crop_y = torch.arange(crop_size[1], device=flow.device) + crop_y1.reshape(-1, 1).to(device=flow.device)  # [B, W']
        # # crop_x, crop_y = torch.meshgrid(torch.arange(crop_size[0], device=flow.device) + crop_x1.reshape(-1, 1).to(device=flow.device), torch.arange(crop_size[1], device=flow.device) + crop_y1.reshape(-1, 1).to(device=flow.device), indexing='ij')
        # crop_x = crop_x.view(-1, crop_size[0], 1).expand(batch, crop_size[0], crop_size[1])  # [B, H', W']
        # crop_y = crop_y.view(-1, 1, crop_size[1]).expand(batch, crop_size[0], crop_size[1])  # [B, H', W']

        # cropped_flow[:, :, top_left_x:top_left_x+crop_size[0], top_left_y:top_left_y+crop_size[1]] = flow[:, :, crop_x1:crop_x1+crop_size[0], crop_y1:crop_y1+crop_size[1]]

        # crop_coords[:, : , top_left_x:top_left_x+crop_size[0], top_left_y:top_left_y+crop_size[1]] = torch.stack([crop_x, crop_y], dim=0).float()  # [B, 2, H', W'] 
        # crop_coords[:, 0, :, :] = crop_coords[:, 0, :, :] / (height - 1)  # normalize to [0, 1]
        # crop_coords[:, 1, :, :] = crop_coords[:, 1, :, :] / (width - 1)
        
        perception_result = torch.cat([flow_downsample, cropped_flow, crop_coords, depth_downsample], dim=1)  # [B, 8, H', W']

        return perception_result
    
    def _perception_advanced(self, flow, depth, crop_midpoint, downsample_size):
        '''
        Args:
            flow: torch.tensor [B, 2, H, W] 
            depth: torch.tensor [B, 2, H, W] :the continuous 2 frame depth
            downsample_size: tuple/list (width', height')(H', W') : the size of the downsampled flow and depth
        Returns:
            perception_result: [B, 8, H', W'] 
            (8 channels where the first 2 channels are the downsample flow, 
            the next 2 channels are the cropped flow, 
            the next 2 channels are the coordinates of cropping area, 
            and the last 2 channels are the downsample depth)
        '''
        flow_downsample = torch.nn.functional.interpolate(flow, size=downsample_size, mode='bilinear', align_corners=False)
        depth_downsample = torch.nn.functional.interpolate(depth, size=downsample_size, mode='bilinear', align_corners=False)
        batch, _, height, width = flow.shape

        crop_coords = torch.zeros((batch, 2, self.crop_size[0], self.crop_size[1]), device=flow.device)  # [B, 2, H', W']
        cropped_flow = torch.zeros((batch, 2, self.crop_size[0], self.crop_size[1]), device=flow.device)  # [B, 2, H', W']

        if isinstance(crop_midpoint[0], torch.Tensor):
            crop_midpoint = (crop_midpoint[0].cpu(), crop_midpoint[1].cpu()) 
        else:
            crop_midpoint = (torch.tensor(crop_midpoint[0]), torch.tensor(crop_midpoint[1])) 
        crop_x1 = crop_midpoint[0] - self.crop_size[0] // 2
        crop_x1 = crop_x1.clamp(0, height - 1)
        crop_x2 = crop_midpoint[0] + self.crop_size[0] // 2
        crop_x2 = crop_x2.clamp(0, height - 1)
        crop_y1 = crop_midpoint[1] - self.crop_size[1] // 2
        crop_y1 = crop_y1.clamp(0, width - 1)
        crop_y2 = crop_midpoint[1] + self.crop_size[1] // 2
        crop_y2 = crop_y2.clamp(0, width - 1)
        top_left_x = self.crop_size[0] // 2 - (crop_midpoint[0] - crop_x1)
        top_left_y = self.crop_size[1] // 2 - (crop_midpoint[1] - crop_y1)

        down_right_x = top_left_x + (crop_x2 - crop_x1)
        down_right_y = top_left_y + (crop_y2 - crop_y1)

        seq_x = torch.arange(self.crop_size[0]).unsqueeze(0)
        seq_y = torch.arange(self.crop_size[1]).unsqueeze(0)

        top_left_x = top_left_x.reshape(-1, 1)
        top_left_y = top_left_y.reshape(-1, 1)
        down_right_x = down_right_x.reshape(-1, 1)
        down_right_y = down_right_y.reshape(-1, 1)

        index_x = ((seq_x >= top_left_x) & (seq_x < down_right_x)).reshape(-1, self.crop_size[0], 1)  # [B, H', 1]
        index_y = ((seq_y >= top_left_y) & (seq_y < down_right_y)).reshape(-1, 1, self.crop_size[1])  # [B, 1, W']

        index_map = index_x & index_y # [B, H', W']
        index_map = index_map.unsqueeze(1).expand_as(cropped_flow) # [B, 2, H', W']

        seq_x = torch.arange(height).unsqueeze(0)
        seq_y = torch.arange(width).unsqueeze(0)

        crop_x1 = crop_x1.reshape(-1, 1)
        crop_y1 = crop_y1.reshape(-1, 1)
        crop_x2 = crop_x2.reshape(-1, 1)
        crop_y2 = crop_y2.reshape(-1, 1)

        crop_index_x = ((seq_x >= crop_x1) & (seq_x < crop_x2)).reshape(-1, height, 1)  # [B, H, 1]
        crop_index_y = ((seq_y >= crop_y1) & (seq_y < crop_y2)).reshape(-1, 1, width)  # [B, 1, W]

        crop_index_map = crop_index_x & crop_index_y # [B, H, W]
        crop_index_map = crop_index_map.unsqueeze(1).expand_as(flow) # [B, 2, H, W]

        cropped_flow[index_map] = flow[crop_index_map]
        cropped_flow = torch.nn.functional.interpolate(cropped_flow, size=downsample_size, mode='bilinear', align_corners=False)
        
        perception_result = torch.cat([flow_downsample, cropped_flow, depth_downsample], dim=1)  # [B, 5, H', W']

        return perception_result
        
    

    def obs_to_input(self, observation, downsample_size=(60, 80), crop_midpoint=(240, 320)):
        '''
        Args:
            obs: list of [B, C, H, W] : the original observation, which contains the RGB image and depth image #of the last two frames
        Returns:
            input_tensor: torch.tensor [B, 8, H', W'] :the input for the actor and critic
        '''
        if crop_midpoint is None:
            crop_midpoint = (observation.shape[2] // 2, observation.shape[3] // 2)
        last_rgb_image = observation[:, 0:3, :, :]
        rgb_image = observation[:, 3:6, :, :]
        depth_image = observation[:, 6:7, :, :]
        n = last_rgb_image.shape[0]
        flow = torch.zeros((n, 2, last_rgb_image.shape[2], last_rgb_image.shape[3]), device=self.device)  # [B, 2, H, W]
        for i in range(n):
            flow[i:i+1] = self._image_to_optical(last_rgb_image[i:i+1], rgb_image[i:i+1]) # (1, 2, H, W)
        # normalize flow
        flow[:, 0, :, :] = (flow[:, 0, :, :] - torch.min(flow[:, 0, :, :])) / (torch.max(flow[:, 0, :, :]) - torch.min(flow[:, 0, :, :]))
        flow[:, 1, :, :] = (flow[:, 1, :, :] - torch.min(flow[:, 1, :, :])) / (torch.max(flow[:, 1, :, :]) - torch.min(flow[:, 1, :, :]))
        
        # input_tensor = self._perception(flow, depth_image, crop_midpoint, crop_size)
        input_tensor = self._perception_advanced(flow, depth_image, crop_midpoint, downsample_size)
        # input_tensor = torch.cat([flow, depth_image], dim=1)  # [B, 3, H, W]
        # input_tensor = nn.AdaptiveAvgPool2d(crop_size)(input_tensor)  # [B, 3, H', W']
        return input_tensor
    
    def _find_max_risk_area(self, risk_map, h, w):
        """
        Finds the h x w area with the maximum sum in a risk map.
        
        Args:
            risk_map (np.ndarray): A 2D array of shape (H, W) or (B, 1, H, W) with values 0-1. In general, B == 1.
            h (int): Height of the target area.
            w (int): Width of the target area.
            
        Returns:
            tuple: (focus_row, focus_col) focus_row and focus_col are the coordinates of the mid point of the best window. 
        """
        risk_map = risk_map.cpu().numpy()
        if risk_map.ndim == 4:
            risk_map = risk_map.squeeze()
        if risk_map.ndim == 2:
            risk_map = risk_map[np.newaxis, :, :]  # Add batch dimension

        B, H, W = risk_map.shape
        
        # Check if the requested window fits in the map
        if h > H or w > W:
            raise ValueError("Window size (h, w) is larger than the map (H, W)")

        # 1. Compute the Integral Image (Summed Area Table)
        # We pad with 0 to handle the top and left edges easily
        integral = np.zeros((B, H + 1, W + 1))
        
        # integral[y, x] = sum of rectangle from (0,0) to (y-1, x-1)
        integral[:, 1:, 1:] = np.cumsum(np.cumsum(risk_map, axis=1), axis=2)

        max_sum = np.zeros(B)  # To store max sum for each batch
        best_r, best_c = np.zeros(B, dtype=int), np.zeros(B, dtype=int)

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
                
                current_sum = (integral[:, r, c] 
                            - integral[:, r - h, c] 
                            - integral[:, r, c - w] 
                            + integral[:, r - h, c - w])

                # 4. Update maximum
                update_index = current_sum > max_sum
                max_sum[update_index] = current_sum[update_index]
                best_r[update_index] = r - h // 2
                best_c[update_index] = c - w // 2

        return best_r, best_c

    def _get_crop_action_by_risk_map(self, depth_map):
        B, H, W = depth_map.shape
        labels = self.risk_perception.generate_labels(depth_map, None)
        risk_map = labels['pixel_risk']
        # crop_size = (self.crop_size[0] // 8, self.crop_size[1] // 8)  # use smaller crop size for action selection
        focus_h, focus_w = self._find_max_risk_area(risk_map, self.crop_size[0], self.crop_size[1])
        crop_action = np.zeros((B, 2))
        # map focus coord from [0, H - 1] into [-1, 1]
        # crop_action[:, 0] = 2 / (self.crop_size[0] - 1) * focus_h - 1
        # crop_action[:, 1] = 2 / (self.crop_size[1] - 1) * focus_w - 1
        crop_action[:, 0] = 2 / (H - 1) * focus_h - 1
        crop_action[:, 1] = 2 / (W - 1) * focus_w - 1

        return crop_action
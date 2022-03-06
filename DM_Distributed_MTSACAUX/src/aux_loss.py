import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from distributional import create_p_categorical
from model_utils import HeadBase, ResNetBlock, NoisyLinear
from modules import Conv2d

from functools import partial


# ---

def dist(a, b):
    batch = a.shape[0]
    assert batch == b.shape[0]
    return F.pairwise_distance(a.view(batch, -1), b.view(batch, -1))


# ---

class LossBase(HeadBase):
    def __init__(self, name, tid, args, action_space, conv_output_size):
        super().__init__(args, action_space)
        self.class_name = self.__class__.__name__
        self.name = name
        self.tid = tid
        self.conv_output_size = conv_output_size


class DiverseDynamicLoss(LossBase):
    # """
    # 正向动力学模型辅助任务
    # """
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, args.state_dim + 10)

    def forward(self, state, action, next_state):
        temp = torch.cat([action, state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), next_state)


class MyInverseDynamicLoss(LossBase):
    # """
    # 逆向动力学模型辅助任务
    # """
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, args.state_dim + 10)

    def forward(self, state, next_state, actions):
        temp = torch.cat([actions, next_state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), state)


# class M

class InverseDynamicLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_h = nn.Linear((args.state_dim + 10) * 2, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, action_space)

    def forward(self, feat1, feat2, actions):
        actions, _ = torch.max(actions, dim=1, keepdim=True)
        x = torch.cat([feat1, feat2], dim=1)
        a = self.fc_z(F.relu(self.fc_h(x)))
        return F.cross_entropy(a, actions.squeeze(1).long(), reduction='none')


class CategoricalRewardLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_z = nn.Linear(args.state_dim + 10, args.hidden_size)
        self.hc_z = nn.Linear(args.hidden_size, action_space)

    def forward(self, state, actor,critic,critic_optim,actor_optim,actions):
        # Backward compatibility, use values read in config file
        attack_epsilon = 0.075
        attack_stepsize = 0.0075
        attack_iteration = 1
        dtype = state.dtype
        state = torch.tensor(state, requires_grad=True)  # convert to tensor
        # ori_state = self.normalize(state.data)
        ori_state_tensor = torch.tensor(state.clone().detach(), dtype=torch.float32)
        # random start
        state = torch.tensor(state, dtype=torch.float32) + ori_state_tensor  # normalized
        # self.attack_epsilon = 0.1
        state_ub = ori_state_tensor + attack_epsilon
        state_lb = ori_state_tensor - attack_epsilon
        for _ in range(attack_iteration):
            state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
            action,_ = actor(state)

            qval,_ = critic.forward(ori_state_tensor, action)
            loss = torch.mean(qval)
            loss.backward()
            adv_state = state - attack_stepsize * state.grad.sign()
            # adv_state = self.normalize(state) + 0.01 * state.grad.sign()
            state = torch.min(torch.max(adv_state, state_lb), state_ub)
            # state =  torch.max(torch.min(adv_state, self.state_max), self.state_min)
        critic_optim.zero_grad()
        actor_optim.zero_grad()
        raction = self.hc_z(self.fc_z(state))
        return F.mse_loss(raction, actions)

    def to_np(self,t):
        return t.cpu().detach().numpy()


class CategoricalIntensityLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        linear = '-l' in name
        if linear:
            print('Linear CI')
            self.net = nn.Linear(self.conv_output_size, action_space * self.args.intensity_atoms)
        else:
            self.net = nn.Sequential(
                nn.Linear(args.state_dim + 10, args.hidden_size), nn.ReLU(),
                nn.Linear(args.hidden_size, action_space)
            )

        self.distrib = create_p_categorical(a=0, b=84, n=args.intensity_atoms, sigma=args.intensity_sigma)

    def intensity(self, x1, x2):
        # assert x1.min() >= 0 and x1.max() <= 1
        diff = dist(x1.mean(1), x2.mean(1)).squeeze()
        # assert torch.all(diff <= 84)
        return diff

    def forward(self, x, actions, x1, x2):
        # with torch.no_grad():
        #     intensities = self.intensity(x1, x2)
        #     intensities = self.distrib(intensities.squeeze())
        i = self.net(x).view(-1, self.action_space)
        i = F.log_softmax(i, dim=1)
        # i = i[range(self.args.batch_size), actions]
        return -torch.sum(i * actions, 1)


class DiscountModel(HeadBase):
    def __init__(self, args, action_space):
        super().__init__(args, action_space)
        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

    def q(self, hv, ha, log=False):
        v = self.fc_z_v(hv).view(-1, 1, self.atoms)
        a = self.fc_z_a(ha).view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        if log:
            q = F.log_softmax(q, dim=2)
        else:
            q = F.softmax(q, dim=2)
        return q

    def forward(self, x, log=False, model=None):
        out = model(x, log=log, return_tuple=True)
        return self.q(out['hv'], out['ha'], log=log)


class DiscountLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, 1)

    def forward(self, state, actions,reward):
        temp = torch.cat([actions, state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), reward)


class MomentChangesLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        self.hidden_layer_size = args.hidden_size

        self.lstm = nn.LSTM(action_space, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size, 1)

        # 初始化隐含状态及细胞状态C，hidden_cell变量包含先前的隐藏状态和单元状态
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

        super().__init__(name, tid, args, action_space, args.conv_output_size)
        self.fc_h = nn.Linear(args.state_dim + 10 + action_space,args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size,args.state_dim + 10)

    def forward(self, state, actions, next_state):
        temp = torch.cat([actions, next_state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), state)


def get_loss_by_name(name):
    if name == 'inverse_dynamic' or name == 'id':
        """
        hidden size
        """
        return InverseDynamicLoss
    elif name == 'categorical_reward' or name == 'cr':
        """
        hidden size 
        reward_atoms
        reward_sigma
        """
        return CategoricalRewardLoss
    elif name == 'moment_changes' or name == 'mc':
        """
        hidden size
        device
        """
        return MomentChangesLoss
    elif 'categorical_intensity' in name or 'ci' in name:
        """
        hidden size
        intensity_atoms
        intensity_sigma
        """
        return CategoricalIntensityLoss
    elif 'discount' in name or 'dsc' in name:
        return DiscountLoss

    elif "MyInverseDynamicLoss" == name:
        return MyInverseDynamicLoss
    elif "DiverseDynamicLoss" == name:
        return DiverseDynamicLoss
    else:
        raise NotImplementedError


def get_aux_loss(name, *args):
    return get_loss_by_name(name)(name, *args)

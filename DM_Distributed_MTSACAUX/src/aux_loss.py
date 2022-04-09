import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import HeadBase


def dist(a, b):
    batch = a.shape[0]
    assert batch == b.shape[0]
    return F.pairwise_distance(a.view(batch, -1), b.view(batch, -1))


class LossBase(HeadBase):
    def __init__(self, name, tid, args, action_space):
        super().__init__(args, action_space)
        self.class_name = self.__class__.__name__
        self.name = name
        self.tid = tid


class DiverseDynamicLoss(LossBase):
    # """
    # ������ѧģ�͸�������
    # """
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_h = nn.Linear(state_dim + num_task + action_space, hidden_size)
        self.fc_z = nn.Linear(hidden_size, state_dim + 10)

    def forward(self, state, action, next_state):
        temp = torch.cat([action, state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), next_state)


class MyInverseDynamicLoss(LossBase):
    # """
    # ������ѧģ�͸�������
    # """
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_h = nn.Linear(state_dim + num_task + action_space, hidden_size)
        self.fc_z = nn.Linear(hidden_size, state_dim + num_task)

    def forward(self, state, next_state, actions):
        temp = torch.cat([actions, next_state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), state)


class InverseDynamicLoss(LossBase):
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_h = nn.Linear((state_dim + num_task) * 2, hidden_size)
        self.fc_z = nn.Linear(hidden_size, action_space)

    def forward(self, feat1, feat2, actions):
        actions, _ = torch.max(actions, dim=1, keepdim=True)
        x = torch.cat([feat1, feat2], dim=1)
        a = self.fc_z(F.relu(self.fc_h(x)))
        return F.cross_entropy(a, actions.squeeze(1).long(), reduction='none')


class AttackRewardLoss(LossBase):
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_z = nn.Linear(state_dim + num_task, hidden_size)
        self.hc_z = nn.Linear(hidden_size, action_space)

    def forward(self, state, actor, critic, critic_optim, actor_optim, actions):
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
            action, _ = actor(state)

            qval, _ = critic.forward(ori_state_tensor, action)
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

    def to_np(self, t):
        return t.cpu().detach().numpy()


class MomentChangesLoss(LossBase):
    def __init__(self, name, tid, args, action_space):
        self.hidden_layer_size = args.hidden_size

        self.lstm = nn.LSTM(action_space, args.hidden_size)
        self.linear = nn.Linear(args.hidden_size, 1)

        # ��ʼ������״̬��ϸ��״̬C��hidden_cell����������ǰ������״̬�͵�Ԫ״̬
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

        super().__init__(name, tid, args, action_space)
        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, args.state_dim + 10)

    def forward(self, state, actions, next_state):
        temp = torch.cat([actions, next_state], dim=1)
        return F.mse_loss(self.fc_z(self.fc_h(temp)), state)


class RewardAttack(LossBase):

    def __init__(self, name, tid, args, action_space, reward_scale, alpha, next_log_probs):
        super().__init__(name, tid, args, action_space)
        self.fc_z = nn.Linear(args.state_dim + 10, args.hidden_size)
        self.hc_z = nn.Linear(args.hidden_size, action_space)

        self.reward_scale = reward_scale

        self.alpha = alpha

        self.next_log_probs = next_log_probs

    def forward(self, critic_value, rewards, state, actions):
        temp = torch.cat([actions, state], dim=1)
        aux_critic_loss = F.mse_loss(self.fc_z(self.fc_h(temp)), state)
        y = self.reward_scale * rewards * (-1) + self.gamma * (
            aux_critic_loss - self.alpha * self.next_log_probs
        )
        return F.mse_loss(y, critic_value)


def get_loss_by_name(name):
    if name == 'InverseDynamic':
        # Ԥ��action����������
        return InverseDynamicLoss
    # elif name == 'MomentChanges':
    #     # ˲ʱ�仯������³������
    #     return MomentChangesLoss
    elif name == "MyInverseDynamicLoss":
        # ������ѧģ�ͣ���������
        return MyInverseDynamicLoss
    elif name == "DiverseDynamicLoss":
        # ������ѧģ�ͣ���������
        return DiverseDynamicLoss
    elif name == "RewardAttack":
        # ģ�͹���������³������
        return RewardAttack


def get_aux_loss(name, *args):
    return get_loss_by_name(name)(name, *args)

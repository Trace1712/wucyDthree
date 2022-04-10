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


class robustReawardLoss(LossBase):
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_z = nn.Linear(state_dim + num_task, hidden_size)
        self.hc_z = nn.Linear(hidden_size, action_space)

        self.attack_epsilon = 0.075
        self.attack_stepsize = 0.0075
        self.attack_iteration = 2

    def forward(self, state, action):
        # ori_state = self.normalize(state.data)
        ori_state_tensor = torch.tensor(state.clone().detach(), dtype=torch.float32)
        # random start
        criterion = torch.nn.MSELoss()
        state = torch.tensor(state, dtype=torch.float32, requires_grad=True)

        for i in range(self.attack_iteration):
            gt_action = self.fc_z(state)
            loss = -criterion(action, gt_action)
            self.zero_grad()
            loss.backward()
            adv_state = state - self.attack_alpha * state.grad.sign()
            state = torch.min(torch.max(adv_state, ori_state_tensor - self.attack_epsilon),
                              ori_state_tensor + self.attack_epsilon)

        return F.mse_loss(gt_action, action)

    def to_np(self, t):
        return t.cpu().detach().numpy()


class MomentChangesLoss(LossBase):
    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)

        self.lstm = nn.LSTM(state_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, action_space)

        # ��ʼ������״̬��ϸ��״̬C��hidden_cell����������ǰ������״̬�͵�Ԫ״̬
        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

        self.fc_h = nn.Linear(args.state_dim + 10 + action_space, args.hidden_size)
        self.fc_z = nn.Linear(args.hidden_size, args.state_dim + 10)

    def forward(self, state, actions, next_state):
        # lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # # lstm������ǵ�ǰʱ�䲽������״̬ht�͵�Ԫ״̬ct�Լ����lstm_out
        # # ����lstm�ĸ�ʽ�޸�input_seq����״����Ϊlinear�������
        # predictions = self.linear(lstm_out)
        # return predictions[-1]  # ����predictions�����һ��Ԫ��
        # temp = torch.cat([actions, next_state], dim=1)
        # return F.mse_loss(self.fc_z(self.fc_h(temp)), state)
        pass


class RewardAttackLoss(LossBase):

    def __init__(self, name, tid, args, action_space, state_dim, hidden_size, num_task):
        super().__init__(name, tid, args, action_space)
        self.fc_z = nn.Linear(state_dim + num_task, hidden_size)
        self.hc_z = nn.Linear(hidden_size, action_space)

        self.reward_scale = args.reward_scale

    def forward(self, critic_value, rewards, state, actions, alpha, next_log_probs):
        temp = torch.cat([actions, state], dim=1)
        aux_critic_loss = F.mse_loss(self.fc_z(self.fc_h(temp)), state)
        y = self.reward_scale * rewards * (-1) + self.gamma * (
            aux_critic_loss - alpha * next_log_probs
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
        return RewardAttackLoss
    elif name == "robustReawardLoss":
        # ģ�͹���״̬��³������
        return robustReawardLoss


def get_aux_loss(name, *args):
    return get_loss_by_name(name)(name, *args)

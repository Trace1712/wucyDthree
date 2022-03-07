import argparse
import datetime

from player import Player
from learner import Learner
import ray
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="run flags")
    parser.add_argument('--seed', default="0", type=int)
    parser.add_argument('--cfg', default="../../cfg/Distributed_MTSAC_cfg_humanoid.json")
    parser.add_argument('--classs', default="walker")
    parser.add_argument('--dir', default="model/")
    parser.add_argument('--run_id', default='16203653-s0')
    parser.add_argument('--cuda_index', type=int, default=None)

    args = parser.parse_args()

    cfg_path = args.cfg
    train_classes = args.classs
    run_id = datetime.datetime.now().strftime('%d%H%M%S')
    if train_classes == "walker":
        train_tasks = ['walk', 'run', 'stand']

    is_train = True  ############## you need to set
    # is_train = False ############## you need to set

    num_cpus = 2  ############## you need to set
    num_gpus = 0  ############## you need to set
    Player = ray.remote(num_cpus=1, num_gpus=0)(Player)  ############## you need to set
    Learner = ray.remote(num_cpus=1, num_gpus=0)(Learner)  ############## you need to set
    ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    ############## you need to set ##############
    # task_distributions = [[0, 2, 5], [1, 3, 7], [4, 6, 8, 9]] # 3 players
    # task_distributions = [[0, 2], [5, 7], [1, 3, 4], [6, 8, 9]] # 4 players
    task_distributions = [[0], [1], [2]]  # 4 players
    # task_distributions = [[0, 2], [5, 7], [1, 3], [6, 8], [4, 9]] # 5 players
    # task_distributions = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]] # 10 players
    ############## you need to set ##############

    networks = []
    aux = ["cr"]
    if is_train:
        for task_idx_list in task_distributions:
            networks.append(
                Player.remote(
                    train_classes,
                    train_tasks,
                    cfg_path,
                    task_idx_list,
                    eval_episode_idx=10,
                    run_id=run_id
                )
            )
        networks.append(Learner.remote(train_classes, train_tasks, cfg_path, save_period=20000, action_dim=4, aux_lst=aux))
        print('Learner added')
    else:
        task_idx_list = [4]  ############## you need to set
        for _ in range(1):
            networks.append(
                Player.remote(
                    train_classes,
                    train_tasks,
                    cfg_path,
                    task_idx_list,
                    train_mode=False,
                    trained_model_path='saved_models/MT10_Distributed_MTSAC/checkpoint_3300000.tar',
                    write_mode=False,
                    render_mode=True,
                    eval_episode_idx=400
                )
            )

    ray.get([network.run.remote() for network in networks])
    ray.shutdown()

#!/usr/bin/env python3

import random
import numpy as np
from agent import PPOAgent
import torch
import matplotlib
matplotlib.use('Agg')
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
import signal
import sys
import gazebo_cooperative_navigation_adjust_pheromone_ir_env as gazebo_env
import csv  # CSVモジュールをインポート
# GPUが使える場合はGPUを使う
from datetime import datetime
import os

def set_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   #　例外処理
    if len(sys.argv) < 2:
        print("Usage: python3 copperative_obstacle_evaluation.py [model_path]")
        sys.exit(0) 

    # 引数からモデルのパスを取得
    model_path = sys.argv[1]

    device = torch.device("cpu")

    total_run = 20 # 実験の試行回数
    plot_interval = 10  # 10イテレーションごとにグラフを保存
    num_env = 1
    seed_value = 1023
    # 実験結果を格納するリスト
    results = []
    set_seeds(seed_value)

    # 引数からモデルのパスを取得

    agent = PPOAgent(env_name="Evaluate_"+gazebo_env.env_name,
                    n_iteration=total_run, 
                    n_states=gazebo_env.n_states, # IR only 
                    action_bounds=[-1, 1], 
                    n_actions=3, # 線形速度と角速度
                    actor_lr=3e-4, 
                    critic_lr=3e-4, 
                    gamma=0.99, 
                    gae_lambda=0.95, 
                    clip_epsilon=0.2, 
                    buffer_size=100000, 
                    batch_size=512,
                    epoch=3,
                    collect_step=256,
                    entropy_coefficient=0.01, 
                    device=device)


    agent.load_weights(model_path)
    agent.set_to_eval_mode()
    env = gazebo_env.GazeboEnvironment(id=0)
    

    for run in range(total_run):
        print("+++++++++++++++++++  evaluation : {}++++++++++++++".format(run))

        step_count = 0

        state = env.reset(seed=random.randint(0,100000))
        logger_action_means = []
        done = [False] * env.robot_num
        # 1エピソードを格納するリスト
        total_reward = [0] * env.robot_num
        total_steps = [0] * env.robot_num
        task_time = [0] * env.robot_num
        done_category = [0] * env.robot_num
        string_done_category = [None] * env.robot_num
        while not all(done): # すべてのエージェントが終了するまで
            state_actor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_mean, _ = agent.actor(state_actor)
            action = np.atleast_2d(action_mean.cpu().numpy())

            next_state, reward, terminated, _ , info = env.step(action)

            for i in range(env.robot_num):
                if done[i] is True:
                    state[i] = [0] * env.n_states
                    continue

                total_steps[i] += 1
                total_reward[i] += reward[i]
            
                state[i] = next_state[i]
                if i == 0:
                    logger_action_means.append(action[i])
                    agent.logger.angle_to_goals_history.append(info[i]["angle_to_goal"])
                    agent.logger.pheromone_mean_history.append(info[i]["pheromone_mean"])
                    agent.logger.pheromone_left_history.append(info[i]["pheromone_left_value"])
                    agent.logger.pheromone_right_history.append(info[i]["pheromone_right_value"])
                    agent.logger.ir_left_history.append(info[i]["ir_left_value"])
                    agent.logger.ir_right_history.append(info[i]["ir_right_value"])
                    

                if terminated[i]:
                    env.stop_robot(i)
                    done[i] = True
                    task_time[i] = info[i]["task_time"]
                    done_category[i] = info[i]["done_category"] # 0: reach goal, 1: collision, 2: time up
                    string_done_category[i] = "reach goal" if done_category[i] == 0 else "collision" if done_category[i] == 1 else "time up"






        action_T_means = np.array(logger_action_means).T.tolist()
        for i in range(agent.n_actions):
            for action_mean in action_T_means[i]:
                agent.logger.action_means_history[i].append(action_mean)

        agent.logger.plot_action_graph(run, agent.n_actions)
        agent.logger.clear_action_logs()
        for i in range(env.robot_num):
            print("total steps: {}, task time: {:.3f}, total reward: {:.3f}, done category: {}".format(total_steps[i], task_time[i], total_reward[i], string_done_category[i]))                
            # 結果をリストに追加
            results.append([run, total_steps[i], task_time[i], total_reward[i], done_category[i]])
    # CSVファイルに結果を書き出す
    dir_name = f"./ppo_Evaluate_" + gazebo_env.env_name + f"/{agent.start_time}"
    # ディレクトリを作成
    os.makedirs(dir_name, exist_ok=True)
    filename = f'{dir_name}/results.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # ヘッダーを書き込む
        writer.writerow(['Run', 'Total Steps', 'Task Time', 'Total Reward', 'Done Category'])
        # データを書き込む
        writer.writerows(results)

def exit(signal, frame):
    print("\nCtrl+C detected. Saving training data and exiting...")    
    sys.exit(0)


if __name__ == '__main__':
    main()

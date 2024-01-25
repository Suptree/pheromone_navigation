import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

import os

class Logger:
    def __init__(self, dir_name, n_actions):
        # log
        # dir_name は 環境名+日付+時間
        self.dir_name = dir_name

        # csvファイルを保存するディレクトリを作成
        self.csv_dir_path = f"{self.dir_name}/training_csv"
        os.makedirs(self.csv_dir_path, exist_ok=True)

        # 行動のグラフを保存するディレクトリを作成
        self.action_graph_dir_path = f"{self.dir_name}/action_graph"
        os.makedirs(self.action_graph_dir_path, exist_ok=True)

        # 学習経過のグラフを保存するディレクトリを作成
        self.training_graph_dir_path = f"{self.dir_name}/training_graph"
        os.makedirs(self.training_graph_dir_path, exist_ok=True)

        ## 報酬ログ
        self.total_rewards_history = []
        self.total_baseline_rewards_history = []
        self.iteration_per_reward_average = []
        ## アクターとクリティックの損失ログ
        self.losses_actors = []
        self.losses_critics = []
        ## アドバンテージのログ
        self.advantage_history = []
        self.advantage_mean_history = []
        self.advantage_variance_history = []
        ## エントロピーのログ
        self.entropies_history = []
        self.entropy_mean_history = []
        ## 学習率のログ
        self.lr_actor_history = []
        self.lr_critic_history = []
        ## アクションの平均と標準偏差のログ
        self.action_means_history = [[] for _ in range(n_actions)]
        self.action_stds_history = [[] for _ in range(n_actions)]
        self.actions_history = [[] for _ in range(n_actions)]
        self.angle_to_goals_history = []
        self.pheromone_mean_history = []
        self.pheromone_history = []
        self.pheromone_left_history = []
        self.pheromone_right_history = []
        self.ir_left_history = []
        self.ir_right_history = []

        self.step_count_history = []

    def plot_graph(self, iteration):
        plt.figure(figsize=(30, 18))
        # 累積報酬のプロット
        plt.subplot(3, 4, 1)
        reward_window_size = 10
        plt.plot(self.total_rewards_history, label="reward", color="green")
        plt.plot(
            self.compute_moving_average(
                self.total_rewards_history, window_size=reward_window_size
            ),
            label="moving reward",
            color="red",
        )
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend(loc='upper left')
        plt.grid(True)

        # 累積報酬のプロット
        plt.subplot(3, 4, 2)
        reward_window_size = 10
        plt.plot(self.total_baseline_rewards_history, label="reward", color="green")
        plt.plot(
            self.compute_moving_average(
                self.total_baseline_rewards_history, window_size=reward_window_size
            ),
            label="moving reward",
            color="red",
        )
        plt.title("Episode Baseline Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend(loc='upper left')
        plt.grid(True)

        
        # アクターの損失のプロット
        plt.subplot(3, 4, 3)
        plt.plot(self.losses_actors)
        plt.title("Actor Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)

        # クリティックの損失のプロット
        plt.subplot(3, 4, 4)
        plt.plot(self.losses_critics)
        plt.title("Critic Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True)

        # アドバンテージの平均のプロット
        plt.subplot(3, 4, 5)
        plt.plot(self.advantage_mean_history)
        plt.title("Average Advantage")
        plt.xlabel("Iteration")
        plt.ylabel("Average Advantage")
        plt.grid(True)

        # アドバンテージの分散のプロット
        plt.subplot(3, 4, 6)
        plt.plot(self.advantage_variance_history)
        plt.title("Variance of Advantage")
        plt.xlabel("Iteration")
        plt.ylabel("Variance")
        plt.grid(True)
        # エントロピーの平均のプロット
        plt.subplot(3, 4, 7)
        plt.plot(self.entropy_mean_history)
        plt.title("Entropy")
        plt.xlabel("Iteration")
        plt.ylabel("Entropy")
        plt.grid(True)
        # 学習率のプロット
        plt.subplot(3, 4, 8)
        plt.plot(self.lr_actor_history, label="actor", color="red")
        plt.plot(self.lr_critic_history, label="critic", color="blue")
        plt.title("Learning Rate")
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.legend(loc='upper left')
        plt.grid(True)

        plt.subplot(3, 4, 9)
        # カテゴリごとに色を設定
        colors = {0: 'green', 1: 'red', 2: 'blue'}
        labels = {0: 'Goal', 1: 'Collision', 2: 'Timeout'}
        # エピソード番号のカウンタを初期化
        episode_counter = 0

        # 各エピソードのステップカウントをプロット
        for i, (cat, step) in enumerate(self.step_count_history):
            plt.bar(i, step, color=colors[cat], label=labels[cat] if i == 0 else "")
        plt.title("Total Steps by Done Category")
        plt.xlabel("Episode")
        plt.ylabel("Total Steps")
        plt.ylim(-10, 400)
        plt.legend(loc='upper left')
        plt.grid(True)

        # 保存
        plt.tight_layout()
        filename = f"{self.training_graph_dir_path}/training_data_{iteration}.png"
        plt.savefig(filename)
        plt.close()

    def plot_action_graph(self, iteration, n_actions):

        # n_actions の値に基づいて新しい Figure を作成
        n_rows = n_actions+3  # アクションの数と角度のために1行追加 # フェロモンでさらに1行追加
        n_cols = 2  # 平均とサンプルのグラフと標準偏差のグラフのために2列

        plt.figure(figsize=(16, 6 * n_rows))  # Figure のサイズを調整
        for i in range(n_actions):
            # 平均とサンプルのグラフ
            plt.subplot(n_rows, n_cols, 2 * i + 1)
            plt.scatter(range(len(self.actions_history[i])), self.actions_history[i], label="sample", color="lime")
            plt.plot(self.action_means_history[i], label="mean", color="red")
            plt.title(f"Action {i} - Means and Samples")
            plt.xlabel("Step")
            plt.ylabel("Action Values")
            plt.ylim(-1.1, 1.1)
            plt.grid(True)
            plt.legend(loc='upper left')

            # 標準偏差のグラフ
            plt.subplot(n_rows, n_cols, 2 * i + 2)
            plt.plot(self.action_stds_history[i], label="std", color="blue")
            plt.title(f"Action {i} - Standard Deviations")
            plt.xlabel("Step")
            plt.ylabel("Action Standard Deviation")
            plt.grid(True)
            plt.legend(loc='upper left')

        # 角度の履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows - 5)  # 新しいサブプロット位置を設定
        plt.plot(self.angle_to_goals_history, label="Angle to Goal", color="purple")
        plt.title("Robot Angle to Goal")
        plt.xlabel("Step")
        plt.ylabel("Angle (degrees)")
        plt.legend(loc='upper left')
        plt.ylim(-200.0, 200.0)
        plt.grid(True)

        # フェロモン平均値の履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows - 4)  # 新しいサブプロット位置を設定
        plt.plot(self.pheromone_mean_history, label="Pheromone Average Value", color="green")
        plt.title("Pheromone Average Value")
        plt.xlabel("Step")
        plt.ylabel("Pheromone Average Value")
        # plt.ylim(-0.1, 1.1)
        plt.legend(loc='upper left')
        plt.grid(True)

        # フェロモン平均値の履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows - 3)  # 新しいサブプロット位置を設定
        plt.plot(self.pheromone_left_history, label="Left Value", color="blue")
        plt.plot(self.pheromone_right_history, label="Right Value", color="orange")
        plt.title("Pheromone Left and Right Average Value")
        plt.xlabel("Step")
        plt.ylabel("Pheromone Average Value")
        # plt.ylim(-0.1, 1.1)
        plt.legend(loc='upper left')
        plt.grid(True)

        # フェロモンの右-左の差分の履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows - 2)  # 新しいサブプロット位置を設定
        plt.plot(np.array(self.pheromone_right_history) - np.array(self.pheromone_left_history), label="Right - Left", color="red")
        plt.title("Pheromone Right - Left")
        plt.xlabel("Step")
        plt.ylabel("Pheromone Right - Left")
        # plt.ylim(-1.1, 1.1)
        plt.legend(loc='upper left')
        plt.grid(True)

        # IRの履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows - 1)  # 新しいサブプロット位置を設定
        plt.plot(self.ir_left_history, label="Left Value", color="blue")
        plt.plot(self.ir_right_history, label="Right Value", color="orange")
        plt.title("IR Left and Right Average Value")
        plt.xlabel("Step")
        plt.ylabel("IR Average Value")
        plt.ylim(-0.1, 0.4)
        plt.legend(loc='upper left')
        plt.grid(True)

        # IRの右-左の差分の履歴のプロット
        plt.subplot(n_rows, n_cols, n_cols * n_rows)  # 新しいサブプロット位置を設定
        plt.plot(np.array(self.ir_right_history) - np.array(self.ir_left_history), label="Right - Left", color="red")
        plt.title("IR Right - Left")
        plt.xlabel("Step")
        plt.ylabel("IR Right - Left")
        plt.ylim(-0.4, 0.4)
        plt.legend(loc='upper left')
        plt.grid(True)


        # グラフを表示
        plt.tight_layout()

        # 必要に応じてファイルに保存
        filename = f"{self.action_graph_dir_path}/action_plots_{iteration}.png"
        plt.savefig(filename)
        plt.close()


    def save_csv(self):
        # 異なるデータカテゴリのファイル名を定義
        rewards_filename = f"{self.csv_dir_path}/episode_rewards.csv"
        losses_filename = f"{self.csv_dir_path}/actor_critic_losses.csv"
        metrics_filename = f"{self.csv_dir_path}/training_metrics.csv"
        learning_rate_filename = f"{self.csv_dir_path}/learning_rate.csv"
        step_count_filename = f"{self.csv_dir_path}/step_count.csv"

        # Episode Rewards のデータフレームを作成して保存
        rewards_df = pd.DataFrame({
            "Episode Rewards": self.total_rewards_history,
            "Baseline Rewards": self.total_baseline_rewards_history,
        })
        self.save_dataframe_to_csv(rewards_df, rewards_filename)

        # Actor Loss と Critic Loss のデータフレームを作成して保存
        losses_df = pd.DataFrame({
            "Actor Loss": self.losses_actors,
            "Critic Loss": self.losses_critics
        })
        self.save_dataframe_to_csv(losses_df, losses_filename)

        # その他のトレーニングメトリクスのデータフレームを作成して保存
        metrics_df = pd.DataFrame({
            "Average Advantage": self.advantage_mean_history,
            "Variance of Advantage": self.advantage_variance_history,
            "Entropy": self.entropy_mean_history,
            # "Actor Learning Rate": self.lr_actor_history,
            # "Critic Learning Rate": self.lr_critic_history
        })
        self.save_dataframe_to_csv(metrics_df, metrics_filename)

        # Actor と Critic の学習率のデータフレームを作成して保存
        learning_rate_df = pd.DataFrame({
            "Actor Learning Rate": self.lr_actor_history,
            "Critic Learning Rate": self.lr_critic_history
        })
        self.save_dataframe_to_csv(learning_rate_df, learning_rate_filename)

        # ステップカウント情報をデータフレームに変換
        step_count_df = pd.DataFrame(self.step_count_history, columns=["Done Category", "Total Steps"])
        self.save_dataframe_to_csv(step_count_df, step_count_filename)
    # def save_dataframe_to_csv(self, dataframe, filename):
    #     # 既存の CSV ファイルがある場合は読み込む
    #     if os.path.exists(filename):
    #         existing_df = pd.read_csv(filename)
    #         updated_df = pd.concat([existing_df, dataframe]).drop_duplicates().reset_index(drop=True)
    #     else:
    #         updated_df = dataframe

    #     # CSV ファイルに保存
    #     updated_df.to_csv(filename, index=False)
    #     print(f"Data saved to {filename}")

    def save_dataframe_to_csv(self, dataframe, filename):
        # 既存の CSV ファイルがある場合は読み込む
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)

            # 既存のデータの長さと新しいデータの長さを比較
            len_existing = len(existing_df)
            len_new = len(dataframe)

            # 新しいデータの末尾から必要な数の行を追加
            if len_new > len_existing:
                diff_df = dataframe.tail(len_new - len_existing)
                updated_df = pd.concat([existing_df, diff_df]).reset_index(drop=True)
            else:
                # 新しいデータが既存のデータより短いか同じ長さの場合、更新は不要
                updated_df = existing_df
        else:
            updated_df = dataframe

        # CSV ファイルに保存
        updated_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
            # 移動平均を計算
    def compute_moving_average(self, history, window_size=10):
        moving_average = []
        for i in range(len(history)):
            start_idx = max(0, i - window_size + 1)
            moving_average.append(np.mean(history[start_idx : i + 1]))
        return moving_average
    
    def calucurate_advantage_mean_and_variance(self):
        # アドバンテージの平均と分散を計算
        adv_mean = np.mean(self.advantage_history)
        adv_variance = np.var(self.advantage_history)

        # ヒストリーリストに追加
        self.advantage_mean_history.append(adv_mean)
        self.advantage_variance_history.append(adv_variance)
        self.advantage_history = []
    def calucurate_entropy_mean(self):
        # エントロピーの平均を計算
        entropy_mean = np.mean(self.entropies_history)

        # ヒストリーリストに追加
        self.entropy_mean_history.append(entropy_mean)
        self.entropies_history = []

    def load_and_merge_csv_data(self, dir_path):
        # 異なるデータカテゴリのファイル名を定義
        rewards_filename = f"{dir_path}/episode_rewards.csv"
        losses_filename = f"{dir_path}/actor_critic_losses.csv"
        metrics_filename = f"{dir_path}/training_metrics.csv"
        learning_rate_filename = f"{dir_path}/learning_rate.csv"
        step_count_filename = f"{dir_path}/step_count.csv"

        # Episode Rewards のデータを読み込んで統合
        if os.path.exists(rewards_filename):
            rewards_df = pd.read_csv(rewards_filename)
            self.total_rewards_history.extend(rewards_df["Episode Rewards"].dropna().tolist())
            self.total_baseline_rewards_history.extend(rewards_df["Baseline Rewards"].dropna().tolist())
        else:
            print(f"{rewards_filename} does not exist.")

        # Actor Loss と Critic Loss のデータを読み込んで統合
        if os.path.exists(losses_filename):
            losses_df = pd.read_csv(losses_filename)
            self.losses_actors.extend(losses_df["Actor Loss"].dropna().tolist())
            self.losses_critics.extend(losses_df["Critic Loss"].dropna().tolist())
        else:
            print(f"{losses_filename} does not exist.")
        # その他のトレーニングメトリクスのデータを読み込んで統合
        if os.path.exists(metrics_filename):
            metrics_df = pd.read_csv(metrics_filename)
            self.advantage_mean_history.extend(metrics_df["Average Advantage"].dropna().tolist())
            self.advantage_variance_history.extend(metrics_df["Variance of Advantage"].dropna().tolist())
            self.entropy_mean_history.extend(metrics_df["Entropy"].dropna().tolist())
        else:
            print(f"{metrics_filename} does not exist.")
        # Actor と Critic の学習率のデータを読み込んで統合
        if os.path.exists(learning_rate_filename):
            learning_rate_df = pd.read_csv(learning_rate_filename)
            self.lr_actor_history.extend(learning_rate_df["Actor Learning Rate"].dropna().tolist())
            self.lr_critic_history.extend(learning_rate_df["Critic Learning Rate"].dropna().tolist())
        else:
            print(f"{learning_rate_filename} does not exist.")

        # ステップカウント情報を読み込んで統合
        if os.path.exists(step_count_filename):
            step_count_df = pd.read_csv(step_count_filename)
            self.step_count_history = list(zip(step_count_df["Done Category"], step_count_df["Total Steps"]))
        else:
            print(f"{step_count_filename} does not exist.")

    def clear_action_logs(self):
        self.action_means_history = [[] for _ in range(len(self.action_means_history))]
        self.actions_history = [[] for _ in range(len(self.actions_history))]
        self.angle_to_goals_history = []
        self.pheromone_mean_history = []
        self.pheromone_left_history = []
        self.pheromone_right_history = []
        self.ir_left_history = []
        self.ir_right_history = []
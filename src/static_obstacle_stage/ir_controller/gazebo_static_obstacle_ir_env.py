import random
import numpy as np
import tf
import math
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty as EmptyMsg
from geometry_msgs.msg import Pose

import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import DeleteModel
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import time
import os
import slackweb
env_name = "Gazebo_Static_Obstacle_IR"
n_states = 13
robot_num = 1
n_actions = 2

class GazeboEnvironment:
    n_states = 13
    
    def __init__(self, id):
        self.id = id
        
        # 固定パラメータ
        self.n_states = 13
        self.robot_num = 1
        self.robot_radius = 0.04408
        self.max_linear_velocity = 0.2
        self.max_angular_velocity = 1.0


        # 原点座標
        self.origin_x = float((int(self.id) % 4) * 20.0) 
        self.origin_y = float(int(int(self.id) / 4) * 20.0)
        
        # ロボットIDを決定
        self.robot_id = [None] * self.robot_num
        for i in range(self.robot_num):
            self.robot_id[i] = self.id * self.robot_num + i

        # ロボットの名前を決定
        self.robot_name = [None] * self.robot_num
        for i in range(self.robot_num):
            self.robot_name[i] = f"hero_{self.robot_id[i]}"

        # step return の初期化
        self.state = [None] * self.robot_num
        self.reward = [None] * self.robot_num
        self.done = [False] * self.robot_num
        self.info = [None] * self.robot_num

        # 非同期更新
        ## Gazeboシミュレータから取得
        self.robot_position = [None] * self.robot_num
        self.robot_linear_velocity = [None] * self.robot_num
        self.robot_angular_velocity = [None] * self.robot_num
        self.robot_angle = [None] * self.robot_num
        ## フェロモンを使用しないので固定値
        self.pheromone_value = [[0] * 9] * self.robot_num
        ## IRセンサから取得
        self.laser_value = [None] * self.robot_num

        # ゴールの位置
        self.goal_pos_x = [None] * self.robot_num
        self.goal_pos_y = [None] * self.robot_num
        self.prev_distance_to_goal = [None] * self.robot_num
        self.max_distance_to_goal = [None] * self.robot_num

        # 静的障害物の位置
        self.obstacle_num = 4
        self.obstacle = [
            # (self.origin_x + 0.0,    self.origin_y + 0.4),
            # (self.origin_x + 0.0,    self.origin_y + (-0.4)),
            # (self.origin_x + 0.4,    self.origin_y + 0.0),
            # (self.origin_x + (-0.4), self.origin_y + 0.0)
        ]
        # 障害物のSDFファイルのパス
        obstacle_path = os.environ.get("STATIC_OBSTACLE_PATH")
        self.obstacle_sdf = open(obstacle_path, "r").read()


        # RobotのLEDの色
        self.robot_color = ["CYEAN"] * self.robot_num

        # ロボットごとのタスク終了フラグ
        self.is_collided = [False] * self.robot_num # 衝突したかどうか
        self.is_goal = [False] * self.robot_num   # ゴールしたかどうか
        self.is_timeout = [False] * self.robot_num # タイムアウトしたかどうか

        # ROSのノードの初期化
        rospy.init_node(f'gazebo_environment_{self.id}', anonymous=True, disable_signals=True)

        # ロボットをコントロールするためのパブリッシャの設定
        ## 速度、角速度のパブリッシャの設定
        self.cmd_vel_pub = [None] * self.robot_num
        for i in range(self.robot_num):
            self.cmd_vel_pub[i] = rospy.Publisher(f'/{self.robot_name[i]}/cmd_vel', Twist, queue_size=1)
        ## LEDの色のパブリッシャの設定
        self.pub_led = [None] * self.robot_num
        for i in range(self.robot_num):
            self.pub_led[i] = rospy.Publisher(f'/{self.robot_name[i]}/led', ColorRGBA, queue_size=1)

        # # Gazeboのモデル(Robot)の状態を設定するためのサービスの設定
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        
        # Gazeboのモデル(Robot)を削除するためのサービスの設定
        rospy.wait_for_service("/gazebo/delete_model")
        self.delete_model_service = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        
        # Gazeboのモデル(静的障害物)を追加するためのサービスの設定
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        self.spawn_model_service = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)

        # Gazeboのモデルの状態を取得するためのサブスクライバの設定
        self.gazebo_model_state_sub = rospy.Subscriber(
            '/gazebo/model_states', ModelStates, self.gazebo_model_state_callback)
        
        # ロボットの相対位置を取得するためのサブスクライバの設定
        self.robot_odometry_sub = [None] * self.robot_num
        for i in range(self.robot_num):
            self.robot_odometry_sub[i] = rospy.Subscriber(f'/{self.robot_name[i]}/odom', Odometry, lambda data, robot_id=i: self.odometry_callback(data, robot_id))

        # pheromoneの値を取得するためのサブスクライバの設定
        # self.sub_phero = rospy.Subscriber(
        #     f'/env_{self.id}/pheromone_value', PheromoneMultiArray2, self.pheromone_callback)
        
        # ロボットのIRセンサの値を取得するためのサブスクライバの設定
        self.laser_sub = [None] * self.robot_num
        for i in range(self.robot_num):
            self.laser_sub[i] = rospy.Subscriber(f'/{self.robot_name[i]}/laser', LaserScan, lambda data, robot_id=i: self.laser_callback(data, robot_id))
        
        # pheromoneの値をリセットするためのパブリッシャの設定
        # self.reset_pheromone_pub = rospy.Publisher(f'/env_{self.id}/pheromone_reset_signal', EmptyMsg, queue_size=1)
        
        # pheromoneの値をリセットするためのパブリッシャの設定
        # self.injection_pheromone_pub = rospy.Publisher(f'/env_{self.id}/pheromone_injection', PheromoneInjection, queue_size=1)
        
        # マーカーを表示するためのパブリッシャの設定        
        self.marker_pub = rospy.Publisher(f'/env_{self.id}/visualization_marker', Marker, queue_size=10)

        
        self.last_time = rospy.Time.now()

        # Initialise simulation
        self.reset_timer = rospy.get_time()



    # 環境のステップを実行する
    def step(self, action): # action = [v, w]
        
        # 時間が止まっている場合は停止
        while self.last_time == rospy.Time.now():
            rospy.sleep(0.01)
                    
        # ロボットに速度を設定
        ## 終了したロボットのアクションは停止
        for i in range(self.robot_num):
            if self.done[i]: # 終了したロボットのアクションは考慮しない
                continue
            v, w = action[i]
            
            # vの値を0.0から0.2の範囲に収める
            v = (v+1.0) * 0.1
            
            # vの値を0から0.2の範囲に収める
            v = max(min(v, 0.2), 0.0)
            # wの値を-1.0から1.0の範囲に収める
            w = max(min(w, 1.0), -1.0)
            twist = Twist()
            twist.linear = Vector3(x=v, y=0, z=0)
            twist.angular = Vector3(x=0, y=0, z=w)
            self.cmd_vel_pub[i].publish(twist)

            # pheromone_injection = PheromoneInjection()
            # pheromone_injection.robot_id = self.robot_id[i]
            # pheromone_injection.radius = 0.3
            # self.injection_pheromone_pub.publish(pheromone_injection)

        rospy.sleep(0.1)

        self.state = [None] * self.robot_num
        reward = [None] * self.robot_num
        baseline_reward = [None] * self.robot_num
        info = [None] * self.robot_num
        # アクション後の環境の状態を取得, 衝突判定やゴール判定も行う
        for i in range(self.robot_num):
            
            # 終了したロボットの次の状態はない
            # return None Object.
            if self.done[i]:
                self.state[i] = [0] * self.n_states
                continue

            # 次の状態を取得
            next_state_pheromone_value,\
            next_state_laser_value,\
            next_state_distance_to_goal,\
            next_state_angle_to_goal,\
            next_state_robot_linear_velocity_x,\
            next_state_robot_angular_velocity_z = self.get_next_state(i)
        

            # フェロモンを読み込んだ場合は緑色にする
            if sum(next_state_laser_value) > 0:
                if self.robot_color[i] != "RED":
                    self.robot_color[i] = "RED"
                    color = ColorRGBA()
                    color.r = 255
                    color.g = 0
                    color.b = 0
                    color.a = 255
                    self.pub_led[i].publish(color)
            else: # pheromone == 0
                if self.robot_color[i] != "CYEAN":
                    self.robot_color[i] = "CYEAN"
                    color = ColorRGBA()
                    color.r = 0
                    color.g = 160
                    color.b = 233
                    color.a = 255
                    self.pub_led[i].publish(color)

            # 報酬の計算
            reward[i], baseline_reward[i] = self.calculate_rewards(i, next_state_distance_to_goal, next_state_robot_angular_velocity_z)

            # ロボットごとの終了判定
            self.done[i] = self.is_collided[i] or self.is_goal[i] or self.is_timeout[i]

            if self.is_goal[i]:
                print(f"\033[1;36m[{self.robot_name[i]}]\033[0m : \033[32m///////   GOAL    ///////\033[0m")
            elif self.is_collided[i]:
                print(f"\033[1;36m[{self.robot_name[i]}]\033[0m : \033[38;5;214m/////// COLLISION ///////\033[0m")

            # INFOの実装
            ## タスク終了したときの情報をInfoに格納
            if self.done[i]:

                task_time = rospy.get_time() - self.reset_timer
                if self.is_goal[i]:
                    done_category = 0
                elif self.is_collided[i]:
                    done_category = 1
                else: # self.is_timeout
                    done_category = 2
                info[i] = {"task_time": task_time, "done_category": done_category, "angle_to_goal": math.degrees(next_state_angle_to_goal),
                        "pheromone_mean": np.mean(next_state_pheromone_value),
                        "pheromone_value": next_state_pheromone_value,
                        "pheromone_left_value" : (next_state_pheromone_value[0] + next_state_pheromone_value[3] + next_state_pheromone_value[6])/3.0,
                        "pheromone_right_value" : (next_state_pheromone_value[2] + next_state_pheromone_value[5] + next_state_pheromone_value[8])/3.0,
                        "ir_left_value" : (next_state_laser_value[0] + next_state_laser_value[3] + next_state_laser_value[5])/3.0,
                        "ir_right_value" : (next_state_laser_value[2] + next_state_laser_value[4] + next_state_laser_value[7])/3.0,
                }
            else:
                info[i] = {"task_time": None, "done_category": None, "angle_to_goal": math.degrees(next_state_angle_to_goal),
                        "pheromone_mean": np.mean(next_state_pheromone_value),
                        "pheromone_value": next_state_pheromone_value,
                        "pheromone_left_value" : (next_state_pheromone_value[0] + next_state_pheromone_value[3] + next_state_pheromone_value[6])/3.0,
                        "pheromone_right_value" : (next_state_pheromone_value[2] + next_state_pheromone_value[5] + next_state_pheromone_value[8])/3.0,
                        "ir_left_value" : (next_state_laser_value[0] + next_state_laser_value[3] + next_state_laser_value[5])/3.0,
                        "ir_right_value" : (next_state_laser_value[2] + next_state_laser_value[4] + next_state_laser_value[7])/3.0,
                }

                
            # 終了したとき、停止
            if self.done[i]:
                twist = Twist()
                twist.linear = Vector3(x=0, y=0, z=0)
                twist.angular = Vector3(x=0, y=0, z=0)
                self.cmd_vel_pub[i].publish(twist)

            # 状態の更新
            self.prev_distance_to_goal[i] = next_state_distance_to_goal
            self.state[i] = list(next_state_laser_value) + [next_state_distance_to_goal, math.sin(next_state_angle_to_goal), math.cos(next_state_angle_to_goal), next_state_robot_linear_velocity_x, next_state_robot_angular_velocity_z]
            self.state[i] = self.normalize_state(self.state[i])

        # if self.id == 0:
        #     print(f"\033[1;36m[{self.robot_name[0]}]\033[0m : \033[38;5;214m{self.state[0]}\033[0m")
        return self.state, reward, self.done, baseline_reward, info


    def calculate_rewards(self, robot_index, next_state_distance_to_goal,next_state_robot_angular_velocity_z):
        Rw = -1.0  # angular velocity penalty constant
        Ra = 30.0  # goal reward constant
        Rc = -30.0 # collision penalty constant
        Rt = -0.1  # time penalty
        w_m = 0.8  # maximum allowable angular velocity
        wd_p = 4.0 # weight for positive distance
        wd_n = 6.0 # weight for negative distance

        # アクション後のロボットとゴールまでの距離の差分
        goal_to_distance_diff = 100.0 * ( self.prev_distance_to_goal[robot_index] - next_state_distance_to_goal)
        
        r_g = Ra if self.is_goal[robot_index] else 0 # goal reward
        r_c = Rc if self.is_collided[robot_index] else 0  # collision penalty
        if goal_to_distance_diff > 0:
            r_d = wd_p * goal_to_distance_diff
        else:
            r_d = wd_n * goal_to_distance_diff
        r_w = Rw if abs(next_state_robot_angular_velocity_z) > w_m else 0  # angular velocity penalty
        r_t = Rt
        reward = r_g + r_c + r_d + r_w
        baseline_reward = r_g + r_c + r_d + r_w

        return reward, baseline_reward

    def get_next_state(self, i):

        # ゴールまでの距離
        next_state_distance_to_goal = math.sqrt((self.robot_position[i].x - self.goal_pos_x[i])**2
                                + (self.robot_position[i].y - self.goal_pos_y[i])**2)
        
        # ロボットの現在の体の向きのベクトルとロボットの現在の位置からゴールまでのベクトルのなす角度
        next_state_angle_to_goal = math.atan2(self.goal_pos_y[i] - self.robot_position[i].y,
                                    self.goal_pos_x[i] - self.robot_position[i].x) - self.robot_angle[i]
        
        ## 角度を-πからπの範囲に正規化
        if next_state_angle_to_goal < -math.pi:
            next_state_angle_to_goal += 2 * math.pi
        elif next_state_angle_to_goal > math.pi:
            next_state_angle_to_goal -= 2 * math.pi

        # 衝突判定
        self.is_collided[i] = self.check_collision_to_obstacle(i)

        # ゴール判定
        self.is_goal[i] = self.check_goal(i)

        # タイムアウト判定
        self.is_timeout[i] = rospy.get_time() - self.reset_timer > 40.0

        return self.pheromone_value[i],self.laser_value[i], next_state_distance_to_goal, next_state_angle_to_goal, self.robot_linear_velocity[i].x, self.robot_angular_velocity[i].z
    
    def normalize_state(self, state):
        """
        各状態を0から1の範囲に正規化する関数。

        Returns:
            list: 正規化された状態のリスト。
        """
        normalized_state = []

        # IRセンサー値の正規化
        for i in range(8):  # 先頭の8つの値はIRセンサー値
            normalized_state.append(state[i] / 0.3)

        env_max_distance_to_goal = 2.0 * math.sqrt(2.0)  # 環境の最大ゴールまでの距離
        # ゴールまでの距離の正規化
        normalized_state.append(state[8] / env_max_distance_to_goal)

        # ゴールまでの角度のsinとcosの正規化（0から1の範囲に）
        normalized_sin_angle_to_goal = (state[9] + 1) / 2
        normalized_cos_angle_to_goal = (state[10] + 1) / 2
        normalized_state.append(normalized_sin_angle_to_goal)
        normalized_state.append(normalized_cos_angle_to_goal)

        # 線形速度の正規化
        normalized_linear_velocity = state[11] / self.max_linear_velocity
        normalized_state.append(normalized_linear_velocity)

        # 角速度の正規化
        normalized_angular_velocity = (state[12] + self.max_angular_velocity) / (2.0 * self.max_angular_velocity)
        normalized_state.append(normalized_angular_velocity)

        return normalized_state

    # ゴールに到達したかどうか
    def check_goal(self, self_index):
        
        distance_to_goal =  math.sqrt((self.robot_position[self_index].x - self.goal_pos_x[self_index])**2
                             + (self.robot_position[self_index].y - self.goal_pos_y[self_index])**2)
        if distance_to_goal <= (0.02 + self.robot_radius):
            return True

        return False
    
    def check_collision_to_obstacle(self, self_index):

        # ロボット同士の衝突を検出
        for i in range(self.robot_num):
            if i == self_index:
                continue
            distance_to_robot = math.sqrt((self.robot_position[self_index].x - self.robot_position[i].x)**2
                             + (self.robot_position[self_index].y - self.robot_position[i].y)**2)
            if distance_to_robot <= 2 * self.robot_radius:
                return True
            
        # 障害物との衝突を検出
        for obs in self.obstacle:
            distance_to_obstacle = math.sqrt((self.robot_position[self_index].x - obs[0])**2 + (self.robot_position[self_index].y - obs[1])**2)
            if distance_to_obstacle <= (0.04408 + 0.02):
                return True

        return False
    
    # 環境のリセット
    def reset(self, seed=None):
        rospy.sleep(0.01)
        if seed is not None:
            random.seed(seed)
        
        # ロボットの速度を停止
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        for i in range(self.robot_num):
            try:
                self.cmd_vel_pub[i].publish(twist)
            except rospy.ServiceException as e:
                print("[def reset] : {0}".format(e))
        rospy.sleep(1.0)


        # マーカーを削除
        self.delete_all_markers()
        # print("delete all markers")

        # 静的障害物を削除
        self.delete_static_obstacle()
        # print("delete all static obstacles")

        # ロボットの位置を原点へリセット
        self.set_initialize_robot()
        # print("reset all robots position")

        # ロボットの色をリセット
        self.set_initialize_robots_color()

        ## ロボットの位置がリセットされるまで待機
        rospy.sleep(3.0)

        # ゴールの初期位置を設定
        self.set_random_goal()
        
        # print("reset all goals position")

        # フェロモンマップをリセット

        # 静的障害物の位置をランダムに設定
        # self.set_distance_range_random_static_obstacle()
        # self.set_static_obstacles()
        self.set_random_distance_static_obstacles()
        # print("reset all static obstacles position")

        # 静的障害物を追加
        self.add_static_obstacle()
        # print("add all static obstacles")

        # 静的障害物が再配置されるまで待機
        rospy.sleep(1.0)


        # マーカーを削除
        self.delete_all_markers()
        # print("delete all markers")
        
        # 新しいゴールマーカーを設定
        self.set_goal_marker()
        # print("set all goal markers")
        # 静的障害物のマーカーを追加
        self.set_obstacle_marker()
        # print("set all static obstacles markers")

        # フラグのリセット
        self.is_collided = [False] * self.robot_num
        self.is_goal = [False] * self.robot_num
        self.is_timeout = [False] * self.robot_num
        self.done = [False] * self.robot_num

        # 変数の初期化
        self.reward = [None] * self.robot_num
        self.distance_to_goal = [None] * self.robot_num
        self.angle_to_goal = [None] * self.robot_num
        self.reset_timer = rospy.get_time()
        self.prev_distance_to_goal = [None] * self.robot_num


    
        # 各ロボットの状態を取得
        self.state = [None] * self.robot_num

        for i in range(self.robot_num):
            self.distance_to_goal[i] = math.sqrt((self.robot_position[i].x-self.goal_pos_x[i])**2
                                    + (self.robot_position[i].y-self.goal_pos_y[i])**2)
            
            # previous distance to goal の初期値をidごとの原点からゴールまでの距離に設定
            self.prev_distance_to_goal[i] = self.distance_to_goal[i]

            angle_to_goal = math.atan2(self.goal_pos_y[i] - self.robot_position[i].y,
                                    self.goal_pos_x[i] - self.robot_position[i].x) - self.robot_angle[i]
            # 角度を-πからπの範囲に正規化
            if angle_to_goal < -math.pi:
                angle_to_goal += 2 * math.pi
            elif angle_to_goal > math.pi:
                angle_to_goal -= 2 * math.pi
            self.angle_to_goal[i] = angle_to_goal

            self.state[i] = list(self.laser_value[i]) + [self.distance_to_goal[i], math.sin(self.angle_to_goal[i]), math.cos(self.angle_to_goal[i]), self.robot_linear_velocity[i].x, self.robot_angular_velocity[i].z]
            self.state[i] = self.normalize_state(self.state[i])
        return self.state
    
    def set_random_goal(self):
        # ゴールの位置をランダムに設定
        for i in range(self.robot_num):
            goal_r = 0.8
            goal_radius = 2.0 * math.pi * random.random()

            self.goal_pos_x[i] = self.origin_x + goal_r * math.cos(goal_radius)
            self.goal_pos_y[i] = self.origin_y + goal_r * math.sin(goal_radius)


    # 完了
    def set_goals(self):
        for i in range(self.robot_num):
            # ゴールを他のロボットの位置に設定
            other_robot_index = (i + 1) % self.robot_num  # 他のロボットのインデックスを取得
            self.goal_pos_x[i] = self.robot_position[other_robot_index].x
            self.goal_pos_y[i] = self.robot_position[other_robot_index].y

            # ロボットとゴールの距離の最大値を設定
            self.max_distance_to_goal[i] = math.sqrt((self.robot_position[i].x - self.goal_pos_x[i])**2
                                    + (self.robot_position[i].y - self.goal_pos_y[i])**2)
    
    def set_initialize_robot(self):

        for i in range(self.robot_num):
            state_msg = ModelState()
            state_msg.model_name = self.robot_name[i]
            state_msg.pose.position.x = self.origin_x
            state_msg.pose.position.y = self.origin_y
            state_msg.pose.position.z = 0.04

            # ランダムな角度をラジアンで生成
            yaw = random.uniform(0, 2 * math.pi)

            # 四元数への変換
            qx = 0.0
            qy = 0.0
            qz = math.sin(yaw / 2)
            qw = math.cos(yaw / 2)

            state_msg.pose.orientation.x = 0.0
            state_msg.pose.orientation.y = 0.0
            state_msg.pose.orientation.z = qz
            state_msg.pose.orientation.w = qw
            state_msg.twist.linear.x = 0.0
            state_msg.twist.linear.y = 0.0
            state_msg.twist.linear.z = 0.0
            state_msg.twist.angular.x = 0.0
            state_msg.twist.angular.y = 0.0
            state_msg.twist.angular.z = 0.0

            try:
                self.set_model_state(state_msg)
            except rospy.ServiceException as e:
                print("[def respawn_robots]: {0}".format(e))
    

    
    # 完了
    def set_initialize_robots_color(self):
        for i in range(self.robot_num):
            # ロボットの色をリセット
            self.robot_color[i] = "CYEAN"
            color = ColorRGBA()
            color.r = 0
            color.g = 160
            color.b = 233
            color.a = 255
            try:
                self.pub_led[i].publish(color)
            except rospy.ServiceException as e:
                print("[def set_initialize_robots_color]: {0}".format(e))

    # 完了
    def shutdown(self):
        """
        Shuts down the ROS node.
        """

        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        try:
            for i in range(self.robot_num):
                self.cmd_vel_pub[i].publish(twist)
        except rospy.ServiceException as e:
            print("[def shutdown]: {0}".format(e))


        rospy.signal_shutdown("Closing Gazebo environment")
        rospy.spin()


    # 完了
    def set_goal_marker(self):
        if self.id != 0:
            return
        """
        Set a goal marker in the Gazebo world and Rviz.
        """
        for i in range(self.robot_num):
            # Rviz
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()

            marker.ns = f"goal_{self.robot_id[i]}"
            marker.id = self.robot_id[i]
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = self.goal_pos_x[i]
            marker.pose.position.y = self.goal_pos_y[i]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.1

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self.marker_pub.publish(marker)


    # 完了
    def delete_all_markers(self):
        # すべてのマーカーを削除するためのマーカーメッセージを作成
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        # マーカーをパブリッシュ
        self.marker_pub.publish(delete_marker)

    def stop_robot(self, robot_index):
        # ロボットの速度を停止
        twist = Twist()
        twist.linear = Vector3(x=0, y=0, z=0)
        twist.angular = Vector3(x=0, y=0, z=0)
        try:
            self.cmd_vel_pub[robot_index].publish(twist)
        except rospy.ServiceException as e:
            print("[def stop_robot]: {0}".format(e))

    # =========== 完成 =============
    # Gazeboのモデルの状態を取得するためのコールバック関数
    def gazebo_model_state_callback(self, model_states):
        for i in range(self.robot_num):
            # 各ロボットのインデックスを取得
            robot_index = model_states.name.index(self.robot_name[i])

            # 各ロボットのposeとtwist情報を取得
            pose = model_states.pose[robot_index]
            twist = model_states.twist[robot_index]
            ori = pose.orientation
            angles = tf.transformations.euler_from_quaternion(
                (ori.x, ori.y, ori.z, ori.w))

            # 各ロボットの情報をクラス変数に格納
            self.robot_position[i] = pose.position
            self.robot_angle[i] = angles[2]
            # self.robot_linear_velocity[i] = twist.linear
            # self.robot_angular_velocity[i] = twist.angular

    # pheromoneの値を取得するためのコールバック関数
    def pheromone_callback(self, phero):
        self.pheromone_value[0] = phero.pheromone1.data
        self.pheromone_value[1] = phero.pheromone2.data


    def laser_callback(self, data, robot_id):
        angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
        angle_range = 15 * (math.pi / 180)  # ±5度をラジアンに変換

        min_distances = []
        for base_angle in angles:
            sector_start = base_angle - angle_range
            sector_end = base_angle + angle_range
            distances = self.get_sector_distances(data, sector_start, sector_end)
            if distances:  # データが存在する場合のみ最小値を計算
                min_distance = min(distances)
                min_distance = 0.31 - min_distance  # 距離が近いほど大きな値になるようにする
            else:
                min_distance = 0.0  # すべてのデータが無限大の場合は0.3メートルとする
            min_distances.append(min_distance)

        self.laser_value[robot_id] = min_distances
        # robot_nameとそれに対応したlaser_valueを表示
        # print(f"{self.robot_name[robot_id]} : {self.laser_value[robot_id]}")

    def odometry_callback(self, data, robot_id):
        self.robot_linear_velocity[robot_id] = data.twist.twist.linear
        self.robot_angular_velocity[robot_id] = data.twist.twist.angular

    def get_sector_distances(self, data, start_angle, end_angle):
        num_points = len(data.ranges)
        distances = []
        for i in range(num_points):
            current_angle = data.angle_min + i * data.angle_increment
            if start_angle <= current_angle <= end_angle or start_angle <= current_angle + 2 * math.pi <= end_angle:
                distance = data.ranges[i]
                if distance != float('inf'):  # 無限大のデータは除外
                    distances.append(distance)
        return distances
    
    def delete_static_obstacle(self):
        """ 静的障害物を削除 """
        for i in range(len(self.obstacle)):
            # 障害物の名前
            obstacle_name = f"obs_{self.id}{i+1}"
            # 障害物の削除
            try:
                self.delete_model_service(obstacle_name)
            except rospy.ServiceException as e:
                self.notify_slack(f"[def delete_static_obstacle]: {e}")
                print("[def delete_static_obstacle]: {0}".format(e))
    def set_distance_random_static_obstacle(self):
        """ 静的障害物の位置をランダムに設定 """
        # 静的障害物の位置をランダムに設定
        self.obstacle = []
        for i in range(4):
            while True:
                # 原点からの距離をランダムに設定
                distance = random.uniform(0.4, 0.8)
                # angle = random.uniform(0, 2 * math.pi)  # 角度をランダムに設定
                angle = i * math.pi / 2.0  # 角度をランダムに設定

                obstacle_x = self.origin_x + distance * math.cos(angle)
                obstacle_y = self.origin_y + distance * math.sin(angle)

                # ゴールとの距離を計算
                distance_to_goal = math.sqrt((obstacle_x - self.goal_pos_x) ** 2 + (obstacle_y - self.goal_pos_y) ** 2)

                # ゴールとの距離が0.1以上ならば配置
                if distance_to_goal >= 0.1:
                    self.obstacle.append((obstacle_x, obstacle_y))
                    break

    def set_distance_range_random_static_obstacle(self):
        """ 静的障害物の位置をランダムに設定 """
        # 静的障害物の位置をランダムに設定
        self.obstacle = []
        angle_offset = math.radians(10)  # ±10度の範囲

        for i in range(self.obstacle_num):
            while True:
                # 原点からの距離をランダムに設定
                distance = random.uniform(0.4, 1.1)
                # angle = random.uniform(0, 2 * math.pi)  # 角度をランダムに設定
                base_angle = i * math.pi / 2.0  # 角度をランダムに設定
                
                # 基本の角度からランダムなオフセットを加える
                obstacle_angle = base_angle + random.uniform(-angle_offset, angle_offset)

                obstacle_x = self.origin_x + distance * math.cos(obstacle_angle)
                obstacle_y = self.origin_y + distance * math.sin(obstacle_angle)

                # ロボットとの距離を計算
                too_close_to_goal = False
                too_close_to_robot = False
                for j in range(self.robot_num):
                    distance_to_robot = math.sqrt((obstacle_x - self.robot_position[j].x) ** 2 + (obstacle_y - self.robot_position[j].y) ** 2)
                    if distance_to_robot < 0.1:
                        too_close_to_robot = True
                        break
                    distance_to_goal = math.sqrt((obstacle_x - self.goal_pos_x[j]) ** 2 + (obstacle_y - self.goal_pos_y[j]) ** 2)
                    if distance_to_goal < 0.35:
                        too_close_to_goal = True
                        break
                if too_close_to_robot:
                    continue
                if too_close_to_goal:
                    continue

                # 他の障害物との距離を計算
                too_close_to_other_obstacle = False
                for existing_obstacle in self.obstacle:
                    distance_to_obstacle = math.sqrt((obstacle_x - existing_obstacle[0]) ** 2 + (obstacle_y - existing_obstacle[1]) ** 2)
                    if distance_to_obstacle < 0.1:
                        too_close_to_other_obstacle = True
                        break
                if too_close_to_other_obstacle:
                    continue

                # 条件を満たした場合は障害物を追加
                self.obstacle.append((obstacle_x, obstacle_y))
                break
    def set_static_obstacles(self):
        # 静的障害物の位置
        self.obstacle = [
            (self.origin_x + 0.0,    self.origin_y + 0.4),
            (self.origin_x + 0.0,    self.origin_y + (-0.4)),
            (self.origin_x + 0.4,    self.origin_y + 0.0),
            (self.origin_x + (-0.4), self.origin_y + 0.0)
        ]
    def set_random_distance_static_obstacles(self):
        # 静的障害物の位置のリストを初期化
        self.obstacle = []

        # 8方向の角度（ラジアン）
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]

        # 各方向に対して障害物を設定
        for angle in angles:
            distance = random.uniform(0.4, 0.75)
            x = self.origin_x + distance * math.cos(angle)
            y = self.origin_y + distance * math.sin(angle)
            self.obstacle.append((x, y))
    def add_static_obstacle(self):
        """ 静的障害物を追加 """
        for i, obs in enumerate(self.obstacle):
            # 障害物の名前
            obstacle_name = f"obs_{self.id}{i+1}"
            # 障害物の初期位置
            initial_pose = Pose()
            initial_pose.position.x = obs[0]
            initial_pose.position.y = obs[1]
            initial_pose.position.z = 0.09
            # 障害物の追加
            try:
                self.spawn_model_service(obstacle_name, self.obstacle_sdf, obstacle_name, initial_pose, "world")
            except rospy.ServiceException as e:
                self.notify_slack(f"[def add_static_obstacle]: {e}")
                print("[def add_static_obstacle]: {0}".format(e))    
    def set_obstacle_marker(self):
        if self.id != 0:
            return
        i = self.robot_num # ゴールのマーカーの次のidから開始
        for obs in self.obstacle:
            i = i + 1
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
        
            marker.ns = "obs"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = obs[0]
            marker.pose.position.y = obs[1]
            marker.pose.position.z = 0.02
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.02

            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            self.marker_pub.publish(marker)

    def notify_slack(self, text):
        slack_web_url = os.environ.get('SLACK_WEB_URL')
        slack = slackweb.Slack(url=slack_web_url)
        rl_ros_machine  = os.environ.get('RL_ROS_MACHINE')

        slack.notify(text=f"{rl_ros_machine} : {text}")

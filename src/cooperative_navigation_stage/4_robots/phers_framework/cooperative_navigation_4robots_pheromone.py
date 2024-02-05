#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty
import math
from gazebo_msgs.msg import ModelStates
import numpy as np
import time
from pathlib import Path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf
import sys

from pheromone_navigation.msg import PheromoneMultiArray2
from pheromone_navigation.msg import PheromoneInjection
from pheromone_navigation.srv import ResetPheromone


class PheromoneFramework:
    def __init__(self, id):
        self.id = id

        self.robot_num = 6
        self.robot_radius = 0.04408

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


        # pheromonクラスのインスタンス生成
        self.pheromone = [None] * self.robot_num
        for i in range(self.robot_num):
            self.pheromone[i] = Pheromone(
                grid_map_size=40, resolution=50, evaporation=0.5, diffusion=0.0
            )


        # フェロモンの最大値と最小値を設定
        self.max_pheromone_value = 1.0
        self.min_pheromone_value = 0.0

        # 経過時間
        self.start_time = rospy.get_time()
        # 最後に受け取った model_states のデータを保存する属性
        self.latest_model_states = None
        
        # 更新用のタイマーを設定
        self.update_interval = rospy.Duration(0.1)  # 0.1秒
        self.update_timer = rospy.Timer(self.update_interval, self.update_callback)


        
        # Publisher & Subscriber
        # フェロモン値を送信
        # self.publish_pheromone = rospy.Publisher(
        #     f"/env_{self.id}/pheromone_value",PheromoneMultiArray2 , queue_size=1
        # )
        #各ロボットのフェロモン値を送信
        self.pheromone_pubs = [None] * self.robot_num
        for i in range(self.robot_num):
            self.pheromone_pubs[i] = rospy.Publisher(
                f"/{self.robot_name[i]}/pheromone", Float32MultiArray, queue_size=1
            )

        # 0.1秒間隔で pheromoneCallback を呼び出すタイマーを設定
        self.pheromone_timer_interval = rospy.Duration(0.1)
        self.pheromone_timer = rospy.Timer(self.pheromone_timer_interval, self.pheromone_timer_callback)

        # gazeboの環境上にあるオブジェクトの状態を取得
        # 取得すると, pheromoneCallback関数が呼び出される
        self.subscribe_pose = rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.get_model_states
        )
        # リセット信号のSubscriber
        # self.subscribe_reset = rospy.Subscriber(
        #     f"/env_{self.id}/pheromone_reset_signal", Empty, self.resetPheromoneMap
        # )
            # フェロモンマップリセットサービスのサーバを設定
        self.reset_service = rospy.Service(
            f"/env_{self.id}/reset_pheromone_service",
            ResetPheromone,
            self.reset_pheromone_service_callback
        )
        # インジェクションメッセージのサブスクライバを設定
        self.injection_subscriber = rospy.Subscriber(
            f"/env_{self.id}/pheromone_injection", PheromoneInjection, self.injection_callback
        )
        self.marker_pub = rospy.Publisher(
            f"/env_{self.id}/visualization_marker_array", MarkerArray, queue_size=1
        )
        # マーカーパブリッシュ用のタイマーを設定（1秒間隔）
        self.marker_publish_interval = rospy.Duration(1.0)  # 1秒
        self.marker_publish_timer = rospy.Timer(self.marker_publish_interval, self.marker_publish_callback)
    
    
    def marker_publish_callback(self, event):
        """
        マーカーを定期的にパブリッシュするためのコールバック関数。
        """
        if self.id == 0:
            # self.publish_markers(id=1, pheromone=self.pheromone[1], r=1.0, g=0.5, b=0.0)
            # self.publish_markers(id=0, pheromone=self.pheromone[0], r=0.0, g=1.0, b=0.0)
            # ロボットごとに色変更
            for i in range(self.robot_num):
                if i == 0:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=0.0, g=1.0, b=0.0)
                elif i == 1:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=1.0, g=0.5, b=0.0)
                elif i == 2:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=0.0, g=0.0, b=1.0)
                elif i == 3:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=1.0, g=0.0, b=1.0)
                elif i == 4:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=0.0, g=1.0, b=1.0)
                elif i == 5:
                    self.publish_markers(id=i, pheromone=self.pheromone[i], r=1.0, g=1.0, b=0.0)
            # for i in range(self.robot_num):
            #     self.publish_markers(id=i, pheromone=self.pheromone[i], r=1.0, g=0.5, b=0.0)

    def publish_markers(self, id, pheromone, r, g, b):
        markerArray = MarkerArray()
        marker_id_offset = id * pheromone.num_cell * pheromone.num_cell


        # フェロモン値の最大値と最小値を計算
        # max_pheromone = np.max(pheromone.grid)
        # min_pheromone = np.min(pheromone.grid)
        max_pheromone = 3.0
        min_pheromone = 0.0
        

        # 正方形の範囲のインデックスを計算
        x_min_index, y_min_index = self.posToIndex(self.origin_x - 1.0, self.origin_y - 1.0, pheromone)
        x_max_index, y_max_index = self.posToIndex(self.origin_x + 1.0, self.origin_y + 1.0, pheromone)

        

        for i in range(x_min_index, x_max_index + 1):
            for j in range(y_min_index, y_max_index + 1):
                # インデックスが範囲外の場合はスキップ
                if i < 0 or j < 0 or i >= pheromone.num_cell or j >= pheromone.num_cell:
                    continue
                if pheromone.grid[i][j] == 0.0:
                    continue

                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "pheromone"
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.z = 0.02
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = 1.0 / pheromone.resolution
                marker.scale.y = 1.0 / pheromone.resolution
                marker.scale.z = 0.01

                marker.color.r = r
                marker.color.g = g
                marker.color.b = b

                # フェロモン値を正規化して透明度に反映
                # if max_pheromone > min_pheromone:
                #     marker.color.a = (pheromone.grid[i][j] - min_pheromone) / (max_pheromone - min_pheromone)
                # else:
                #     marker.color.a = 0.0

                threshold = 0.005  # 閾値の設定
                if pheromone.grid[i][j] > threshold:
                    marker.color.a = pheromone.grid[i][j]
                else:
                    continue  # 閾値以下の場合はマーカーを追加しない

                (marker.pose.position.x, marker.pose.position.y) = self.indexToPos(i, j, pheromone)
                marker.id = i * pheromone.num_cell + j + marker_id_offset
                markerArray.markers.append(marker)
        # print("marker array size: {}".format(len(markerArray.markers)))
        self.marker_pub.publish(markerArray)
    # 完了
    def delete_all_markers(self):
        # すべてのマーカーを削除するためのマーカーメッセージを作成
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

    # 座標からフェロモングリッドへ変換
    def posToIndex(self, x, y, pheromone_map):
        x_index = math.floor((x - self.origin_x + ((1.0/ pheromone_map.resolution)/2.0) +  pheromone_map.grid_map_size/2.0) *  pheromone_map.resolution)
        y_index = math.floor((y - self.origin_y + ((1.0/ pheromone_map.resolution)/2.0) + pheromone_map.grid_map_size/2.0) * pheromone_map.resolution)
        if (
            x_index < 0
            or y_index < 0
            or x_index > pheromone_map.num_cell - 1
            or y_index > pheromone_map.num_cell - 1
        ):
            raise Exception("The pheromone matrix index is out of range.")
        return x_index, y_index    # フェロモングリッドから座標へ変換
    
    
    # セルの中央を返す
    def indexToPos(self, x_index, y_index, pheromone_map): 
        x = (x_index -  (pheromone_map.resolution * pheromone_map.grid_map_size / 2.0)) * (1.0 / pheromone_map.resolution) + self.origin_x
        y = (y_index -  (pheromone_map.resolution * pheromone_map.grid_map_size / 2.0)) * (1.0 / pheromone_map.resolution) + self.origin_y

        return x, y
    
    def get_model_states(self, model_states):
        self.latest_model_states = model_states

    def pheromone_timer_callback(self, event):
        try:
            if self.latest_model_states is None:
                return
            model_status = self.latest_model_states
            
            for i, robot_name in enumerate(self.robot_name):
                # フェロモン値を保存するためのリスト
                pheromone_value = [ 0.0 ] * 9
                # ロボットのインデックスを取得
                robot_index = model_status.name.index(robot_name)
                pose = model_status.pose[robot_index]
                pos = pose.position
                ori = pose.orientation
                angles = tf.transformations.euler_from_quaternion(
                    (ori.x, ori.y, ori.z, ori.w))
                theta = angles[2]

                # 他のロボットのフェロモンマップを参照するためすべてのロボットでループ
                for j in range(self.robot_num):
                    if i == j:
                        # 自分自身のフェロモンマップは参照しない
                        continue
                    # j番目のロボットのフェロモンマップを参照
                    angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
                    for k, angle in enumerate(angles):
                        adjusted_angle = theta + angle
                        dir_x = pos.x + self.robot_radius * math.cos(adjusted_angle)
                        dir_y = pos.y + self.robot_radius * math.sin(adjusted_angle)
                        x_index, y_index = self.posToIndex(dir_x, dir_y, self.pheromone[j])
                        pheromone_value[k] += self.pheromone[j].getPheromone(x_index, y_index)
                        if k == 3:
                            # ロボット自身の位置でのフェロモン値を追加
                            x_index, y_index = self.posToIndex(pos.x, pos.y, self.pheromone[j])
                            pheromone_value[k] += self.pheromone[j].getPheromone(x_index, y_index)
                
                # フェロモン値を送信
                pheromone_value_msg = Float32MultiArray()
                pheromone_value_msg.data = pheromone_value
                self.pheromone_pubs[i].publish(pheromone_value_msg)
        except Exception as e:
            rospy.logerr("Error occurred in pheromoneCallback: {}".format(e))
            for _ in range(9):
                pheromone_value_msg.data.append(0.0)
            for i in range(self.robot_num):
                if not rospy.is_shutdown():
                    self.pheromone_pubs[i].publish(pheromone_value_msg)
                


    def pheromoneCallback(self, model_status):
        # pheromone_multi_value = PheromoneMultiArray2()
        # try:
        #     self.latest_model_states = model_status
        #     pheromone_value = [Float32MultiArray(), Float32MultiArray()]

        #     for i, robot_name in enumerate(self.robot_name):
        #         # ロボットのインデックスを取得
        #         robot_index = model_status.name.index(robot_name)
        #         pose = model_status.pose[robot_index]
        #         pos = pose.position
        #         ori = pose.orientation
        #         angles = tf.transformations.euler_from_quaternion(
        #             (ori.x, ori.y, ori.z, ori.w))
        #         theta = angles[2]

        #         # 他のロボットのフェロモンマップを参照するためのインデックス
        #         other_phero_index = (i + 1) % self.robot_num
        #         angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
        #         for j, angle in enumerate(angles):
        #             adjusted_angle = theta + angle
        #             dir_x = pos.x + self.robot_radius * math.cos(adjusted_angle)
        #             dir_y = pos.y + self.robot_radius * math.sin(adjusted_angle)
        #             x_index, y_index = self.posToIndex(dir_x, dir_y, self.pheromone[other_phero_index])
        #             pheromone_value[i].data.append(
        #                 self.pheromone[other_phero_index].getPheromone(x_index, y_index)
        #             )
        #             if j == 3:
        #                 # ロボット自身の位置でのフェロモン値を追加
        #                 x_index, y_index = self.posToIndex(pos.x, pos.y, self.pheromone[other_phero_index])
        #                 pheromone_value[i].data.append(
        #                     self.pheromone[other_phero_index].getPheromone(x_index, y_index)
        #                 )



        # except Exception as e:
        #     rospy.logerr("Error occurred in pheromoneCallback: {}".format(e))
        #     for _ in range(9):  # 8方向 + 自身の座標 = 9個の値
        #         pheromone_multi_value.pheromone1.data.append(0.0)
        #         pheromone_multi_value.pheromone2.data.append(0.0)

        # if not rospy.is_shutdown():
        #     pheromone_multi_value.pheromone1 = pheromone_value[0]
        #     pheromone_multi_value.pheromone2 = pheromone_value[1]
        #     self.publish_pheromone.publish(pheromone_multi_value)
        
        # 各ロボットのフェロモン値を送信
        ## 各ロボットは自分以外のフェロモンマップの合計値を送信
        try:
            self.latest_model_states = model_status
            for i, robot_name in enumerate(self.robot_name):
                # フェロモン値を保存するためのリスト
                pheromone_value = [ 0.0 ] * 9
                # ロボットのインデックスを取得
                robot_index = model_status.name.index(robot_name)
                pose = model_status.pose[robot_index]
                pos = pose.position
                ori = pose.orientation
                angles = tf.transformations.euler_from_quaternion(
                    (ori.x, ori.y, ori.z, ori.w))
                theta = angles[2]

                # 他のロボットのフェロモンマップを参照するためすべてのロボットでループ
                for j in range(self.robot_num):
                    if i == j:
                        # 自分自身のフェロモンマップは参照しない
                        continue
                    # j番目のロボットのフェロモンマップを参照
                    angles = [math.pi/4, 0, -math.pi/4, math.pi/2, -math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4]
                    for k, angle in enumerate(angles):
                        adjusted_angle = theta + angle
                        dir_x = pos.x + self.robot_radius * math.cos(adjusted_angle)
                        dir_y = pos.y + self.robot_radius * math.sin(adjusted_angle)
                        x_index, y_index = self.posToIndex(dir_x, dir_y, self.pheromone[j])
                        pheromone_value[k] += self.pheromone[j].getPheromone(x_index, y_index)
                        if k == 3:
                            # ロボット自身の位置でのフェロモン値を追加
                            x_index, y_index = self.posToIndex(pos.x, pos.y, self.pheromone[j])
                            pheromone_value[k] += self.pheromone[j].getPheromone(x_index, y_index)
                
                # フェロモン値を送信
                pheromone_value_msg = Float32MultiArray()
                pheromone_value_msg.data = pheromone_value
                self.pheromone_pubs[i].publish(pheromone_value_msg)
        except Exception as e:
            rospy.logerr("Error occurred in pheromoneCallback: {}".format(e))
            for _ in range(9):
                pheromone_value_msg.data.append(0.0)
            for i in range(self.robot_num):
                if not rospy.is_shutdown():
                    self.pheromone_pubs[i].publish(pheromone_value_msg)
                


    def resetPheromoneMap(self):
        print(f"Resetting pheromone map {self.id}")
        for i in range(self.robot_num):
            self.pheromone[i].reset()
        self.delete_all_markers()
        # 経過時間
        self.start_time = rospy.get_time()

    def reset_pheromone_service_callback(self, request):
        self.resetPheromoneMap()
        return {'success': True}
    def injection_callback(self, msg):
        try:
            # ロボットIDに基づいてフェロモンを射出
            if self.latest_model_states is None:
                return
            robot_id = msg.robot_id
            pheromone_radius = msg.radius
            robot_index = self.latest_model_states.name.index(f"hero_{robot_id}")
            pose = self.latest_model_states.pose[robot_index]
            pos = pose.position
            # ロボットの角度を取得
            ori = pose.orientation
            angles = tf.transformations.euler_from_quaternion(
                (ori.x, ori.y, ori.z, ori.w))
            theta = angles[2]

            # 対応するロボットのフェロモンマップにフェロモンを射出
            index = robot_id % self.robot_num
            x_index, y_index = self.posToIndex(pos.x, pos.y, self.pheromone[index])
            # self.pheromone[index].injectionCircle(x_index, y_index, self.max_pheromone_value, pheromone_radius)
            # self.pheromone[index].injectionEllipseRotated(x_index, y_index, self.max_pheromone_value, pheromone_radius, pheromone_radius/2, theta)
            self.pheromone[index].injectionCircleAhead(x_index, y_index, self.max_pheromone_value, pheromone_radius, theta, 0.15)
        except Exception as e:
            rospy.logerr("Error in injection_callback: {}".format(e))

    def update_callback(self, event):
        min_pheromone_value = 0.0  # 最小フェロモン値を設定
        max_pheromone_value = 1.0  # 最大フェロモン値を設定
        for pheromone_map in self.pheromone:
            pheromone_map.update(min_pheromone_value, max_pheromone_value)
        
class Pheromone:
    def __init__(self, grid_map_size=0, resolution=0, evaporation=0.0, diffusion=0.0):
        # グリッド地図の生成
        # map size = 1 m * size
        self.grid_map_size = grid_map_size
        # grid cell size = 1 m / resolution
        self.resolution = resolution
        # 1辺におけるグリッドセルの合計個数
        self.num_cell = self.resolution * self.grid_map_size + 1
        # 例外処理 : 一辺におけるグリッドセルの合計個数を必ず奇数でなければならない
        if self.num_cell % 2 == 0:
            raise Exception("Number of cell is even. It needs to be an odd number")
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))

        # 蒸発パラメータの設定
        self.evaporation = evaporation
        if self.evaporation == 0.0:
            self.isEvaporation = False
        else:
            self.isEvaporation = True

        # 拡散パラメータの設定
        self.diffusion = diffusion
        if self.diffusion == 0.0:
            self.isDiffusion = False
        else:
            self.isDiffusion = True

        # Timers
        self.step_timer = rospy.get_time()
        self.injection_timer = rospy.get_time()
        self.save_timer = rospy.get_time()
        self.reset_timer = rospy.get_time()

    # 指定した座標(x, y)からフェロモンを取得
    def getPheromone(self, x, y):
        return self.grid[x, y]

    # 指定した座標(x, y)へフェロモンを配置
    def setPheromone(self, x, y, value):
        self.grid[x, y] = value

    # 正方形の形でフェロモンの射出する
    def injectionSquare(
        self, x, y, pheromone_value, injection_size, max_pheromone_value
    ):
        # 例外処理 : フェロモンを射出するサイズはかならず奇数
        if injection_size % 2 == 0:
            raise Exception("Pheromone injection size must be an odd number.")
        # 現在時刻を取得
        current_time = rospy.get_time()
        # フェロモンを射出する間隔を0.1s間隔に設定
        if current_time - self.injection_timer > 0.1:
            for i in range(injection_size):
                for j in range(injection_size):
                    # 指定した範囲のセルにフェロモンを配置
                    self.grid[
                        x - (injection_size - 1) / 2 + i,
                        y - (injection_size - 1) / 2 + j,
                    ] += pheromone_value
                    # 配置するフェロモン値がフェロモンの最大値より大きければカット
                    if (
                        self.grid[
                            x - (injection_size - 1) / 2 + i,
                            y - (injection_size - 1) / 2 + j,
                        ]
                        >= max_pheromone_value
                    ):
                        self.grid[
                            x - (injection_size - 1) / 2 + i,
                            y - (injection_size - 1) / 2 + j,
                        ] = max_pheromone_value
            # フェロモンを射出した時間を記録
            self.injection_timer = current_time

    def injectionCircle(self, x_index, y_index, max_value, radius):
        radius = int(radius * self.resolution)
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                distance_squared = i**2 + j**2
                if distance_squared <= radius**2:
                    # 中心からの距離を計算
                    distance = np.sqrt(distance_squared)

                    # 中心に近いほど値が大きく、外縁に近いほど値が小さくなるようにする
                    # 例: 線形減衰（max_value から 0 まで線形に減少）
                    value = max_value * (1 - distance / radius)

                    self.grid[x_index + i, y_index + j] = self.grid[x_index + i, y_index + j] + value

    def injectionEllipseRotated(self, x_index, y_index, max_value, major_axis, minor_axis, angle_deg):
        angle_rad = np.radians(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        major_axis = int(major_axis * self.resolution)
        minor_axis = int(minor_axis * self.resolution)

        for i in range(-major_axis, major_axis + 1):
            for j in range(-minor_axis, minor_axis + 1):
                # 回転前の座標における楕円方程式
                distance_squared = (i**2 / major_axis**2) + (j**2 / minor_axis**2)

                if distance_squared <= 1:
                    # 座標系の回転を適用
                    x_rotated = i * cos_angle - j * sin_angle
                    y_rotated = i * sin_angle + j * cos_angle

                    # 回転後の座標を元のグリッド座標系に変換
                    x_grid = int(x_rotated + x_index)
                    y_grid = int(y_rotated + y_index)

                    # 中心からの距離に基づいてフェロモン値を計算
                    distance = np.sqrt(distance_squared)
                    value = max_value * (1 - distance)

                    # グリッド内の値を更新
                    self.grid[x_grid, y_grid] = self.grid[x_grid, y_grid] + value
    def injectionCircleAhead(self, x_index, y_index, max_value, radius, direction_angle, ahead_distance):
        # direction_angle（ラジアン単位）に基づいて進行方向を決定
        ahead_x = ahead_distance * np.cos(direction_angle)
        ahead_y = ahead_distance * np.sin(direction_angle)

        # 新しい射出中心の座標を計算
        new_x_index = x_index + int(ahead_x * self.resolution)
        new_y_index = y_index + int(ahead_y * self.resolution)

        # 円形の射出処理
        radius = int(radius * self.resolution)
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                distance_squared = i**2 + j**2
                if distance_squared <= radius**2:
                    distance = np.sqrt(distance_squared)
                    value = max_value * (1 - distance / radius)
                    self.grid[new_x_index + i, new_y_index + j] = self.grid[new_x_index + i, new_y_index + j] + value
    def update(self, min_pheromone_value, max_pheromone_value):
        # current_time = rospy.get_time()
        update_interval = 0.1

        # 拡散を行うかどうかの判定
        if self.isDiffusion is True:
            # Diffusion
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    self.grid_copy[i, j] += 0.9 * self.grid[i, j]
                    if i >= 1:
                        self.grid_copy[i - 1, j] += 0.025 * self.grid[i, j]
                    if j >= 1:
                        self.grid_copy[i, j - 1] += 0.025 * self.grid[i, j]
                    if i < self.num_cell - 1:
                        self.grid_copy[i + 1, j] += 0.025 * self.grid[i, j]
                    if j < self.num_cell - 1:
                        self.grid_copy[i, j + 1] += 0.025 * self.grid[i, j]
            # 最大と最小を丸める
            for i in range(self.num_cell):
                for j in range(self.num_cell):
                    if self.grid_copy[i, j] < min_pheromone_value:
                        self.grid_copy[i, j] = min_pheromone_value
                    elif self.grid_copy[i, j] > max_pheromone_value:
                        self.grid_copy[i, j] = max_pheromone_value
            # グリッドセルを更新
            self.grid = np.copy(self.grid_copy)
            # 複製したグリッドセルを初期化
            self.grid_copy = np.zeros((self.num_cell, self.num_cell))

        # 蒸発を行うかどうかの判定
        if self.isEvaporation is True:
            # Evaporation
            decay = 2 ** (-update_interval / self.evaporation)
            self.grid *= decay
            # for i in range(self.num_cell):
            #     for j in range(self.num_cell):
            #         self.grid[i, j] = decay * self.grid[i, j]

    def save(self, file_name):
        parent = Path(__file__).resolve().parent
        with open(
            parent.joinpath("pheromone_saved/" + file_name + ".pheromone"), "wb"
        ) as f:
            np.save(f, self.grid)
        # print("The pheromone matrix {} is successfully saved".
        # format(file_name))

    def load(self, file_name):
        parent = Path(__file__).resolve().parent
        with open(parent.joinpath("pheromone_saved/" + file_name + ".npy"), "rb") as f:
            self.grid = np.load(f)
        # print("The pheromone matrix {} is successfully loaded".
        #       format(file_name))
    def reset(self):
        self.grid = np.zeros((self.num_cell, self.num_cell))
        self.grid_copy = np.zeros((self.num_cell, self.num_cell))


if __name__ == "__main__":
    # コマンドライン引数を確認
    if len(sys.argv) < 2:
        print("Please provide the number of parallel environments")
        sys.exit(1)

    # 引数を取得
    num_env = int(sys.argv[1])

    rospy.init_node(f"env_{num_env}_Pheromone_Framework", anonymous=True,)

    node = PheromoneFramework(id=num_env)
    rospy.spin()

#!/bin/bash

# SIGINT (Ctrl+C) を受け取ったときに実行する関数
cleanup() {
    echo "Stopping all ROS nodes..."
    kill -SIGINT ${pids[@]}
    exit
}

# 引数からノードの数を取得
number_of_nodes=$1

# 引数が指定されていない、または数値でない場合はエラーメッセージを表示して終了
if ! [[ "$number_of_nodes" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 <number_of_nodes>"
    exit 1
fi

# SIGINT シグナルを捕捉するためのハンドラを設定
trap cleanup SIGINT

# 各ノードのプロセスIDを格納する配列
declare -a pids

# 指定された数だけノードを起動し、プロセスIDを配列に追加
for (( i=0; i<number_of_nodes; i++ ))
do
    rosrun pheromone_navigation cooperative_navigation_pheromone.py $i &
    pids+=($!)
done

# Ctrl+Cが押されるまで待機
echo "ROS nodes are running. Press Ctrl+C to stop."
while true; do
    sleep 60
done
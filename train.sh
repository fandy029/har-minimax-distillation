#!/bin/bash
# 训练 - 全部数据集（后台运行）
# 用法: ./train.sh [--datasets ds1 ds2] [--versions v1 v2] [--force] [--resume]
cd "$(dirname "$0")"
LOG="./train.log"
nohup python3 scripts/run_train_all.py "$@" >> "$LOG" 2>&1 &
echo "[后台] 训练已启动 (PID=$!)"
echo "查看日志: tail -f $LOG"
echo "查看训练详情: tail -f results/logs/<dataset>_<version>_train.log"
echo "杀进程: ./kill_all.sh"

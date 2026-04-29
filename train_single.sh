#!/bin/bash
# 训练 - 单个数据集（后台运行）
# 用法: ./train_single.sh <dataset> [version] [--resume] [--force]
cd "$(dirname "$0")"
if [ -z "$1" ]; then
    echo "用法: $0 <dataset> [version] [--resume] [--force]"
    echo "数据集: pamap2, kuhar, uci_har, harth, uci_har_new, motionsense, gait, wisdm, motionsense_dm"
    echo "版本:   pure_cnn, v1, v2, v3"
    exit 1
fi
DATASET=$1
VERSION=${2:-v1}
shift 2
LOG="./train_single.log"
nohup python3 scripts/run_train.py "$DATASET" "$VERSION" "$@" >> "$LOG" 2>&1 &
echo "[后台] 训练($DATASET/$VERSION)已启动 (PID=$!)"
echo "查看日志: tail -f $LOG"

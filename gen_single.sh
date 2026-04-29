#!/bin/bash
# 软标签生成 - 单个数据集（后台运行）
# 用法: ./gen_single.sh <dataset> [--ratio 0.40] [--limit 400] [--force]
cd "$(dirname "$0")"
if [ -z "$1" ]; then
    echo "用法: $0 <dataset> [--ratio 0.40] [--limit 400] [--force]"
    echo "可用数据集: pamap2, uci_har, harth, uci_har_new, motionsense, gait, wisdm, motionsense_dm"
    echo "注意: kuhar 请使用 gen_kuhar_parallel.py start"
    exit 1
fi
LOG="./gen_single.log"
nohup python3 scripts/run_gen_soft.py "$@" >> "$LOG" 2>&1 &
echo "[后台] 软标签生成($1)已启动 (PID=$!)"
echo "查看日志: tail -f $LOG"

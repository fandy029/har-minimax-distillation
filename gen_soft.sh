#!/bin/bash
# 软标签生成 - 全部数据集（后台运行）
# 用法: ./gen_soft.sh [--ratio 0.40] [--limit 400] [--force]
cd "$(dirname "$0")"
LOG="./gen_soft.log"
nohup python3 scripts/run_gen_all.py "$@" >> "$LOG" 2>&1 &
echo "[后台] 软标签生成已启动 (PID=$!)"
echo "工作目录: $(pwd)"
echo "启动日志: $LOG"
echo "子进程日志: results/logs/gen_<dataset>.log"
echo "查看启动日志: tail -f $LOG"
echo "杀进程: ./kill_all.sh"

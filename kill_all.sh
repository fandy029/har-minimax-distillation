#!/bin/bash
# 杀掉所有软标签生成和训练进程
# 用法: ./kill_all.sh
cd "$(dirname "$0")"
pkill -f "scripts/gen_soft_labels_unified.py" 2>/dev/null
pkill -f "scripts/gen_kuhar_parallel.py" 2>/dev/null
pkill -f "scripts/run_distill.py" 2>/dev/null
pkill -f "scripts/run_train_all.py" 2>/dev/null
pkill -f "scripts/run_gen_all.py" 2>/dev/null
echo "✅ 已发送终止信号"
echo "确认: ps -ef | grep python3"

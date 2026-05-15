#!/bin/bash
# 从 gait/ 目录运行，适配新的目录结构
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAIT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$GAIT_DIR"

if [ "$1" = "--stop" ]; then
    PIDS=$(pgrep -f "gait_gen.py")
    [ -z "$PIDS" ] && echo "gait: 无进程" || { kill $PIDS 2>/dev/null; echo "gait: 已停止"; }
    exit 0
fi

mkdir -p output/logs output/soft_labels output/checkpoints output/per_class
echo "gait ($@)"

# prepare 只跑一次
python3 soft_label/gait_prepare.py
echo "data ready"

for cls in $(seq 0 3); do
    nohup python3 soft_label/gait_gen.py --class $cls $@ > output/logs/stdout_${cls}.log 2>&1 &
    echo "  class $cls (PID $!)"
    sleep 10
done
echo "4 processes started"

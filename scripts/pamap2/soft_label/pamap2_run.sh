#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAIT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$GAIT_DIR"

if [ "$1" = "--stop" ]; then
    PIDS=$(pgrep -f "pamap2_gen.py")
    [ -z "$PIDS" ] && echo "pamap2: 无进程" || { kill $PIDS 2>/dev/null; echo "pamap2: 已停止"; }
    exit 0
fi
mkdir -p output/logs output/soft_labels output/checkpoints output/per_class
echo "pamap2 ($@)"
python3 soft_label/pamap2_prepare.py
echo "data ready"
for cls in $(seq 0 4); do
    nohup python3 soft_label/pamap2_gen.py --class $cls $@ > output/logs/stdout_${cls}.log 2>&1 &
    echo "  class $cls (PID $!)"
    sleep 10
done
echo "5 processes started"

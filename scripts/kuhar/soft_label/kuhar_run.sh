#!/bin/bash
cd "$(dirname "$0")"
if [ "$1" = "--stop" ]; then
    PIDS=$(pgrep -f "kuhar_gen.py")
    [ -z "$PIDS" ] && echo "kuhar: 无进程" || { kill $PIDS 2>/dev/null; echo "kuhar: 已停止"; }
    exit 0
fi
mkdir -p output/logs output/soft_labels output/checkpoints output/per_class
echo "kuhar () $@"

# prepare 只跑一次
python3 kuhar_prepare.py > /dev/null 2>&1
echo "data ready"

for cls in $(seq 0 17); do
    nohup python3 kuhar_gen.py --class $cls $@ > output/logs/stdout_${cls}.log 2>&1 &
    echo "  class $cls (PID $!)"
    sleep 10
done
echo "18 processes started"

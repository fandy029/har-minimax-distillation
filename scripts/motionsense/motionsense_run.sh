#!/bin/bash
cd "$(dirname "$0")"
if [ "$1" = "--stop" ]; then
    PIDS=$(pgrep -f "motionsense_gen.py")
    [ -z "$PIDS" ] && echo "motionsense: 无进程" || { kill $PIDS 2>/dev/null; echo "motionsense: 已停止"; }
    exit 0
fi
mkdir -p output/logs output/soft_labels output/checkpoints output/per_class
echo "motionsense () $@"

# prepare 只跑一次
python3 motionsense_prepare.py > /dev/null 2>&1
echo "data ready"

for cls in $(seq 0 5); do
    nohup python3 motionsense_gen.py --class $cls $@ > output/logs/stdout_${cls}.log 2>&1 &
    echo "  class $cls (PID $!)"
    sleep 10
done
echo "6 processes started"

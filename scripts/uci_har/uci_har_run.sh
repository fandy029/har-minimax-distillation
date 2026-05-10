#!/bin/bash
cd "$(dirname "$0")"
if [ "$1" = "--stop" ]; then
    PIDS=$(pgrep -f "uci_har_gen.py")
    [ -z "$PIDS" ] && echo "uci_har: 无进程" || { kill $PIDS 2>/dev/null; echo "uci_har: 已停止"; }
    exit 0
fi
mkdir -p output/logs output/soft_labels output/checkpoints output/per_class
echo "uci_har () $@"

# prepare 只跑一次
python3 uci_har_prepare.py > /dev/null 2>&1
echo "data ready"

for cls in $(seq 0 5); do
    nohup python3 uci_har_gen.py --class $cls $@ > output/logs/stdout_${cls}.log 2>&1 &
    echo "  class $cls (PID $!)"
    sleep 10
done
echo "6 processes started"

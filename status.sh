#!/bin/bash
# 查看软标签生成和训练状态
# 用法: ./status.sh
cd "$(dirname "$0")"

TS=$(date '+%Y-%m-%d %H:%M:%S')


echo "=============================================="
echo " $TS 进程状态"
echo "=============================================="
RUNNING=$(ps -ef| grep "python3"| grep -v status.sh | grep -v grep)
if [ -n "$RUNNING" ]; then
    echo "$RUNNING"
else
    echo "无运行中的脚本进程"
fi
echo ""
echo "=============================================="
echo " 软标签进度"
echo "=============================================="
python3 scripts/soft_label_progress.py 2>/dev/null || echo "(无法获取进度)"


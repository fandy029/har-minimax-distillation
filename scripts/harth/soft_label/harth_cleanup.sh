#!/bin/bash
# 删除 KuHar 软标签生成的所有输出
# 用法: bash cleanup.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="$SCRIPT_DIR/output"

if [ -d "$TARGET" ]; then
    echo "将删除: $TARGET"
    echo "包含: soft_labels/ logs/ checkpoints/ per_class/"
    echo ""
    read -p "确认删除? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET"
        echo "已删除"
    else
        echo "取消"
    fi
else
    echo "output/ 目录不存在，无需清理"
fi

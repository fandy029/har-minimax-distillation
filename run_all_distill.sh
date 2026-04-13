#!/bin/bash
# 批量运行所有蒸馏实验

cd /home/fandy/workplace/thesis

DATASETS=("pamap2" "kuhar" "uci_har" "harth" "uci_har_new" "motionsense" "gait" "wisdm" "motionsense_dm")
VERSIONS=("pure_cnn" "v1" "v2" "v3")

LOG_DIR="results"
SOFT_DIR="results/soft_labels"

for dataset in "${DATASETS[@]}"; do
    for version in "${VERSIONS[@]}"; do
        LOG_FILE="${LOG_DIR}/${dataset}_${version}_log.txt"
        
        # 检查是否已经完成（有结果文件）
        RESULT_FILE="${LOG_DIR}/${dataset}_${version}.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "[$(date)] $dataset $version - 已完成，跳过"
            continue
        fi
        
        # 检查软标签是否存在
        SOFT_FILE="${SOFT_DIR}/${dataset}_soft.npy"
        if [ ! -f "$SOFT_FILE" ]; then
            echo "[$(date)] $dataset $version - 软标签不存在: $SOFT_FILE，跳过"
            continue
        fi
        
        echo "[$(date)] 开始: $dataset $version"
        python3 -u run_distill.py "$dataset" "$version" > "$LOG_FILE" 2>&1
        
        if [ -f "$RESULT_FILE" ]; then
            echo "[$(date)] 完成: $dataset $version"
        else
            echo "[$(date)] 失败: $dataset $version"
        fi
    done
done

echo "[$(date)] 所有实验完成"

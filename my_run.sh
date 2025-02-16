#!/bin/bash

# 配置参数
CUDA_VISIBLE_DEVICES="1,2"           # 手动指定可见的 GPU（逗号分隔）
export CUDA_VISIBLE_DEVICES            # 必须导出环境变量

# 自动计算 GPU 数量
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

# 检查 GPU 数量有效性
if [ $NUM_GPUS -eq 0 ]; then
  echo "❌ 错误：CUDA_VISIBLE_DEVICES 未指定有效 GPU 设备！"
  exit 1
fi

# 分布式训练参数
MASTER_ADDR="localhost"
MASTER_PORT="12347"
DATASET_NAME="CUHK-PEDES"

# 训练脚本参数
TRAIN_ARGS=(
  "--name" "iira"
  "--img_aug"
  "--batch_size" "64"
  "--MLM"
  "--dataset_name" "${DATASET_NAME}"
  "--loss_names" "sdm+mlm+id"
  "--root_dir" "/home/cxd/storage/proj/IRRA/_id"
  "--num_epoch" "60"
)

# 通过 torchrun 启动分布式训练
echo "🚀 启动训练：使用 ${NUM_GPUS} 个 GPU（设备 ID: ${CUDA_VISIBLE_DEVICES}）"
torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py "${TRAIN_ARGS[@]}"

# 错误处理
if [ $? -ne 0 ]; then
  echo "⚠️ 训练失败！请检查日志."
  exit 1
fi
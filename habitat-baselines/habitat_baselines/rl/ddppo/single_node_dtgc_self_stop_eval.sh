#!/bin/bash
#SBATCH -p i64m1tga40ue         # 指定GPU队列 i64m1tga40u
#SBATCH -o output_%j.txt  # 指定作业标准输出文件，%j为作业号
#SBATCH -e err_%j.txt    # 指定作业标准错误输出文件
#SBATCH -n 1           # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH --time=08:58:00 

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
export PYTHONPATH=$(pwd)/../../..:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HYDRA_FULL_ERROR=1 
# 设置每个GPU使用16个CPU核心
# CPU_CORES_PER_GPU=1 # 16
TOTAL_GPU=1

# 将总的CPU核心数计算出来
# TOTAL_CPU_CORES=$((CPU_CORES_PER_GPU * TOTAL_GPU))

# set -x

python -u -m habitat-baselines.habitat_baselines.run \
    --config-name=social_nav_v2/dtgc_self_stop_hm3d_eval_with_csv.yaml \
    > evaluation/dtgc5_hmap_self_stop/hm3d/eval-ckpt11.log 2>&1


echo "FINISH"
# OMP_NUM_THREADS=$CPU_CORES_PER_GPU \
    # python -u -m torch.distributed.launch \
    # --use_env \
    # --nproc_per_node $TOTAL_GPU \
    # habitat-baselines/habitat_baselines/run.py \
    # --config-name=social_nav_v2/hm3d_train_map_future_map_refine.yaml \
    # > evaluation/map_future_1_only_sigmoid_bce_s10/hm3d/train.log 2>&1
    # --config-name=social_nav_v2/falcon_hm3d.yaml \
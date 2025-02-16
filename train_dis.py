import os
import os.path as op
import torch
import torch.distributed as dist
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    args = get_args()

    # 手动设置可用 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # 手动指定可用的 GPU 设备
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    num_gpus = len(visible_devices)

    # 设置种子
    set_seed(1 + get_rank())  # 确保每个进程种子不同

    # 分布式训练初始化
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)  # 绑定每个进程到特定 GPU
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()
        args.distributed = True
    else:
        args.local_rank = 0  # 单卡模式
        args.distributed = False

    device = torch.device("cuda", args.local_rank)
    is_master = get_rank() == 0

    # 设置输出路径与日志
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, args.dataset_name, f'{cur_time}_{args.name}')
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training, distributed_rank=get_rank())
    if is_master:
        logger.info(f"Using {num_gpus} GPUs: {visible_devices}")
        logger.info(str(args).replace(',', '\n'))
        save_train_configs(args.output_dir, args)

    # 数据加载器（使用 DistributedSampler）
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args, distributed=args.distributed)

    # 构建模型
    model = build_model(args, num_classes)
    logger.info('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # 分布式数据并行封装
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False
        )

    # 优化器与调度器
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # 模型检查点与评估器
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(val_img_loader, val_txt_loader)

    # 加载断点恢复
    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    # 开始训练
    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

if __name__ == '__main__':
    main()
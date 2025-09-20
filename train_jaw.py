#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PointAttN 牙列数据训练脚本
基于原始train.py，专门用于训练牙列补全模型
"""

import torch.optim as optim
import torch
from utils.train_utils import *
import logging
import math
import importlib
import datetime
import random
import munch
import yaml
import os
import sys
import argparse
from dataset import C3D_h5, PCN_pcd

def setup_logging(work_dir, flag):
    """设置日志"""
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    log_file = os.path.join(work_dir, f'{flag}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def check_data_path(pcnpath):
    """检查数据路径是否正确"""
    required_dirs = [
        os.path.join(pcnpath, 'train'),
        os.path.join(pcnpath, 'test'),
        os.path.join(pcnpath, 'train', 'complete'),
        os.path.join(pcnpath, 'train', 'partial'),
        os.path.join(pcnpath, 'test', 'complete'),
        os.path.join(pcnpath, 'test', 'partial')
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logging.error(f"Required directory not found: {dir_path}")
            return False

    # 检查元数据文件
    metadata_files = [
        os.path.join(pcnpath, 'PCN.json'),
        os.path.join(pcnpath, 'category.txt')
    ]

    for file_path in metadata_files:
        if not os.path.exists(file_path):
            logging.warning(f"Metadata file not found: {file_path}")

    return True

def train():
    """主训练函数"""
    logging.info("Starting PointAttN training for jaw completion")
    logging.info(f"Configuration: {args}")

    # 检查数据路径
    if not check_data_path(args.pcnpath):
        logging.error("Data path validation failed. Please run preprocessing first.")
        return

    # 设置训练指标
    metrics = ['cd_p', 'cd_t', 'cd_t_coarse', 'cd_p_coarse']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    # 加载数据集
    logging.info("Loading datasets...")
    try:
        dataset = PCN_pcd(args.pcnpath, prefix="train")
        dataset_test = PCN_pcd(args.pcnpath, prefix="test")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True  # 确保batch size一致
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers)
    )

    logging.info(f'Train dataset length: {len(dataset)}')
    logging.info(f'Test dataset length: {len(dataset_test)}')
    logging.info(f'Train batches: {len(dataloader)}')
    logging.info(f'Test batches: {len(dataloader_test)}')

    # 设置随机种子
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info(f'Random Seed: {seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 加载模型
    logging.info("Initializing model...")
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()

    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
        logging.info("Applied weight initialization")

    # 设置优化器
    lr = args.lr
    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    if args.optimizer == 'Adagrad':
        optimizer = optimizer(net.module.parameters(), lr=lr, initial_accumulator_value=args.initial_accum_val)
    else:
        betas = args.betas.split(',')
        betas = (float(betas[0].strip()), float(betas[1].strip()))
        optimizer = optimizer(net.module.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)

    logging.info(f"Optimizer: {args.optimizer}, LR: {lr}, Weight Decay: {args.weight_decay}")

    # 加载预训练模型（如果指定）
    if args.load_model:
        if os.path.exists(args.load_model):
            ckpt = torch.load(args.load_model)
            net.module.load_state_dict(ckpt['net_state_dict'])
            logging.info(f"Loaded pretrained model from {args.load_model}")
        else:
            logging.warning(f"Pretrained model not found: {args.load_model}")

    # 开始训练
    logging.info("Starting training loop...")
    best_cd = float('inf')

    for epoch in range(args.start_epoch, args.nepoch):
        # 重置训练损失计量器
        train_loss_meter.reset()
        net.module.train()

        # 学习率衰减
        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # 训练一个epoch
        epoch_start_time = datetime.datetime.now()

        for i, data in enumerate(dataloader):
            optimizer.zero_grad()

            try:
                _, inputs, gt = data
                inputs = inputs.float().cuda()
                gt = gt.float().cuda()

                # 前向传播
                loss, _ = net.module.get_loss(inputs, gt)

                # 反向传播
                loss.backward()
                optimizer.step()

                train_loss_meter.update(loss.item())

                # 打印训练状态
                if i % args.step_interval_to_print == 0:
                    logging.info(
                        f'Epoch [{epoch}/{args.nepoch}] Batch [{i}/{len(dataloader)}] '
                        f'Loss: {loss.item():.6f} Avg: {train_loss_meter.avg:.6f} LR: {lr:.8f}'
                    )

            except Exception as e:
                logging.error(f"Error in training step {i}: {e}")
                continue

        epoch_end_time = datetime.datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time

        logging.info(
            f'Epoch [{epoch}/{args.nepoch}] completed in {epoch_duration}. '
            f'Average Loss: {train_loss_meter.avg:.6f}'
        )

        # 验证
        if epoch % args.epoch_interval_to_val == 0:
            logging.info("Running validation...")

            # 重置验证指标
            for meter in val_loss_meters.values():
                meter.reset()

            net.module.eval()
            with torch.no_grad():
                for data in dataloader_test:
                    try:
                        _, inputs, gt = data
                        inputs = inputs.float().cuda()
                        gt = gt.float().cuda()

                        # 前向传播
                        loss, _ = net.module.get_loss(inputs, gt, eval=True)
                        val_loss_meters['cd_p'].update(loss.item())

                    except Exception as e:
                        logging.error(f"Error in validation: {e}")
                        continue

            current_cd = val_loss_meters['cd_p'].avg
            logging.info(f'Validation CD: {current_cd:.6f}')

            # 保存最佳模型
            if current_cd < best_cd:
                best_cd = current_cd
                best_model_path = os.path.join(args.work_dir, f'best_model_{args.flag}.pth')
                torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cd_loss': current_cd,
                }, best_model_path)
                logging.info(f'New best model saved with CD: {best_cd:.6f}')

        # 定期保存模型
        if epoch % args.epoch_interval_to_save == 0 and epoch > 0:
            model_path = os.path.join(args.work_dir, f'model_{args.flag}_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_meter.avg,
            }, model_path)
            logging.info(f'Model saved at epoch {epoch}')

    logging.info("Training completed!")
    logging.info(f"Best validation CD: {best_cd:.6f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/PointAttN_Jaw.yaml',
                       help='YAML config file')
    argv = parser.parse_args()

    # 加载配置文件
    with open(argv.config, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    args = munch.munchify(args)

    # 设置日志
    setup_logging(args.work_dir, args.flag)

    # 检查CUDA
    if not torch.cuda.is_available():
        logging.error("CUDA is not available!")
        exit(1)

    logging.info(f"Using GPU: {args.device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # 开始训练
    train()
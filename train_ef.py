import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

import albumentations as A
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score  # [CHANGED] indicators 제거

from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter

import shutil
import json

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)

    parser.add_argument('--dataseed', default=2981, type=int)
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    
    # dataset
    parser.add_argument('--dataset', default='busi')
    parser.add_argument('--image_dir')
    parser.add_argument('--mask_dir')
    parser.add_argument('--splits_final', type=str)

    parser.add_argument('--output_dir', default='outputs')

    # optimizer
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    parser.add_argument('--kan_lr', default=1e-2, type=float)
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--cfg', type=str, metavar="FILE")
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')

    config = parser.parse_args()
    return config

# ---------------- AMP scaler (global) ----------------
scaler = torch.cuda.amp.GradScaler()  # [CHANGED]

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda(non_blocking=True)   # [CHANGED]
        target = target.cuda(non_blocking=True) # [CHANGED]

        optimizer.zero_grad(set_to_none=True)   # [CHANGED]

        # compute output with AMP
        with torch.cuda.amp.autocast():         # [CHANGED]
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

        # backward
        scaler.scale(loss).backward()           # [CHANGED]
        scaler.step(optimizer)                  # [CHANGED]
        scaler.update()                         # [CHANGED]

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        pbar.set_postfix(OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ]))
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():     # [CHANGED]
                if config['deep_supervision']:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou, dice, _ = iou_score(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            pbar.set_postfix(OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ]))
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_torch()
    config = vars(parse_args())

    # dataset-specific mask suffix
    dataset_name = config['dataset']
    img_ext = '_0000.nii.gz'
    mask_ext = '.png'
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name in ['glas', 'cvc', 'isic2018', 'isic2017']:
        mask_ext = '.png'
    elif dataset_name == "ngtube":
        mask_ext = ".nii.gz"
    config['mask_ext'] = mask_ext

    if config['name'] is None:
        config['name'] = f"{config['dataset']}_{config['arch']}_{'wDS' if config['deep_supervision'] else 'woDS'}"

    exp_name = config['name']
    output_dir = config['output_dir']
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    print('-' * 20)
    for key in config:
        print(f'{key}: {config[key]}')
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # loss
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    # cuDNN: 입력 크기가 고정이라면 빠름
    cudnn.benchmark = True  # [CHANGED]

    # model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan']
    ).cuda()

    # -------- param groups: only TWO groups (kan vs others) -------- [CHANGED]
    kan_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('layer' in name.lower()) and ('fc' in name.lower()):
            kan_params.append(param)
        else:
            base_params.append(param)

    param_groups = [
        {'params': base_params, 'lr': config['lr'], 'weight_decay': config['weight_decay']},
        {'params': kan_params,  'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']},
    ]

    # optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            param_groups,
            lr=config['lr'],
            momentum=config['momentum'],
            nesterov=config['nesterov'],
            weight_decay=config['weight_decay']
        )
    else:
        raise NotImplementedError

    # scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr']
        )
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config['factor'], patience=config['patience'],
            verbose=1, min_lr=config['min_lr']
        )
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(e) for e in config['milestones'].split(',')],
            gamma=config['gamma']
        )
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # keep a copy of code
    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    # Data
    img_dir = config['image_dir']
    mask_dir = config['mask_dir']

    with open(config["splits_final"]) as f:
        splits = json.load(f)
        train_img_ids = splits[0]["train"]
        val_img_ids   = splits[0]["val"]

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=img_dir, mask_dir=mask_dir,
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform
    )
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=img_dir, mask_dir=mask_dir,
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )

    # dataloader kwargs depending on workers  ------------------------ [CHANGED]
    dl_common = dict(
        batch_size=config['batch_size'],
        pin_memory=True
    )
    if config['num_workers'] > 0:
        dl_common.update(dict(
            num_workers=config['num_workers'],
            persistent_workers=True,
            prefetch_factor=8
        ))
    else:
        dl_common.update(dict(num_workers=0))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, **dl_common
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,   # [CHANGED] 버그 수정
        shuffle=False, drop_last=False, **dl_common
    )

    log = OrderedDict([
        ('epoch', []), ('lr', []),
        ('loss', []), ('iou', []),
        ('val_loss', []), ('val_iou', []), ('val_dice', []),
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0

    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train / validate
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log   = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f' %
              (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        log['epoch'].append(epoch)
        log['lr'].append(current_lrs)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], epoch)
        my_writer.add_scalar('train/iou',  train_log['iou'],  epoch)
        my_writer.add_scalar('val/loss',   val_log['loss'],   epoch)
        my_writer.add_scalar('val/iou',    val_log['iou'],    epoch)
        my_writer.add_scalar('val/dice',   val_log['dice'],   epoch)
        my_writer.add_scalar('val/best_iou_value',  best_iou,  epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

if __name__ == '__main__':
    main()

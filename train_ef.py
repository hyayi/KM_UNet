import argparse, os, random, json, shutil, yaml
from collections import OrderedDict
import numpy as np, pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from tqdm import tqdm
import albumentations as A

import archs, losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from tensorboardX import SummaryWriter


def list_type(s): return [int(a) for a in s.split(',')]

def parse_args():
    p = argparse.ArgumentParser()
    # --- basics ---
    p.add_argument('--name', default=None)
    p.add_argument('--epochs', default=400, type=int)
    p.add_argument('-b', '--batch_size', default=16, type=int)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--output_dir', default='outputs')

    # --- data ---
    p.add_argument('--dataset', default='busi')
    p.add_argument('--image_dir', required=True)
    p.add_argument('--mask_dir',  required=True)
    p.add_argument('--splits_final', type=str, required=True)

    # --- model ---
    p.add_argument('--arch', default='UKAN')
    p.add_argument('--deep_supervision', default=False, type=str2bool)
    p.add_argument('--input_channels', default=3, type=int)
    p.add_argument('--num_classes', default=1, type=int)
    p.add_argument('--input_w', default=256, type=int)
    p.add_argument('--input_h', default=256, type=int)
    p.add_argument('--input_list', type=list_type, default=[128,160,256])
    p.add_argument('--no_kan', action='store_true')

    # --- loss ---
    LOSS_NAMES = losses.__all__ + ['BCEWithLogitsLoss']
    p.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # --- optim ---
    p.add_argument('--optimizer', default='Adam', choices=['Adam','SGD'])
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--momentum', default=0.9, type=float)
    p.add_argument('--nesterov', default=False, type=str2bool)

    # KAN 전용 하이퍼
    p.add_argument('--kan_lr', default=1e-2, type=float)
    p.add_argument('--kan_weight_decay', default=1e-4, type=float)

    # --- scheduler ---
    p.add_argument('--scheduler', default='CosineAnnealingLR',
                   choices=['CosineAnnealingLR','ReduceLROnPlateau','MultiStepLR','ConstantLR'])
    p.add_argument('--min_lr', default=1e-5, type=float)
    p.add_argument('--factor', default=0.1, type=float)
    p.add_argument('--patience', default=2, type=int)
    p.add_argument('--milestones', default='1,2', type=str)
    p.add_argument('--gamma', default=2/3, type=float)
    p.add_argument('--early_stopping', default=-1, type=int)

    return p.parse_args()


def seed_all(seed=1029):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # 입력 크기 고정 시 autotuner로 가속, 변동 크기면 False 권장
    cudnn.benchmark = True   # 고정 256x256이면 보통 이득. :contentReference[oaicite:1]{index=1}
    cudnn.deterministic = False


def build_criterion(name):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss().cuda()
    return losses.__dict__[name]().cuda()


def make_dataloaders(cfg, img_ext='_0000.nii.gz', mask_ext='.png'):
    # 데이터셋별 마스크 확장자
    if cfg['dataset'] == 'busi':
        mask_ext = '_mask.png'
    elif cfg['dataset'] in ['glas','cvc','isic2018','isic2017']:
        mask_ext = '.png'
    elif cfg['dataset'] == 'ngtube':
        mask_ext = '.nii.gz'

    with open(cfg["splits_final"]) as f:
        sp = json.load(f)
        train_ids, val_ids = sp[0]["train"], sp[0]["val"]

    train_tf = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])
    val_tf = A.Compose([
        A.Resize(cfg['input_h'], cfg['input_w']),
        A.Normalize(),
    ])

    train_ds = Dataset(train_ids, cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=train_tf)
    val_ds   = Dataset(val_ids,   cfg['image_dir'], cfg['mask_dir'],
                       img_ext, mask_ext, num_classes=cfg['num_classes'],
                       transform=val_tf)

    dl_common = dict(batch_size=cfg['batch_size'], pin_memory=True)
    if cfg['num_workers'] > 0:
        dl_common.update(dict(num_workers=cfg['num_workers'],
                              persistent_workers=True,  # 에폭간 워커 유지로 시작 지연 감소 :contentReference[oaicite:2]{index=2}
                              prefetch_factor=8))        # 워커당 미리 적재 수 (기본 2) :contentReference[oaicite:3]{index=3}
    else:
        dl_common.update(dict(num_workers=0))

    train_loader = torch.utils.data.DataLoader(
        train_ds, shuffle=True, drop_last=True, **dl_common
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,   shuffle=False, drop_last=False, **dl_common  # ⚠️ 검증은 val_ds 사용(버그 수정)
    )
    return train_loader, val_loader


def build_optimizer(cfg, model):
    # 파라미터 그룹 2개만(kan vs 기타) → 수천 그룹 생성 오버헤드 방지
    kan_params, base_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if ('layer' in n.lower()) and ('fc' in n.lower()):
            kan_params.append(p)
        else:
            base_params.append(p)

    groups = [
        {'params': base_params, 'lr': cfg['lr'],     'weight_decay': cfg['weight_decay']},
        {'params': kan_params,  'lr': cfg['kan_lr'], 'weight_decay': cfg['kan_weight_decay']},
    ]
    if cfg['optimizer'] == 'Adam':
        return optim.Adam(groups)
    return optim.SGD(groups, lr=cfg['lr'], momentum=cfg['momentum'],
                     nesterov=cfg['nesterov'], weight_decay=cfg['weight_decay'])


def build_scheduler(cfg, optimizer):
    if cfg['scheduler'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'], eta_min=cfg['min_lr'])
    if cfg['scheduler'] == 'ReduceLROnPlateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg['factor'],
                                              patience=cfg['patience'], verbose=True, min_lr=cfg['min_lr'])
    if cfg['scheduler'] == 'MultiStepLR':
        return lr_scheduler.MultiStepLR(optimizer,
                milestones=[int(e) for e in cfg['milestones'].split(',')], gamma=cfg['gamma'])
    if cfg['scheduler'] == 'ConstantLR':
        return None
    raise NotImplementedError


def train_one_epoch(cfg, loader, model, criterion, optimizer):
    avg = {'loss': AverageMeter(), 'iou': AverageMeter()}
    model.train()
    pbar = tqdm(total=len(loader))
    for x, y, _ in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # 메모리/성능에 유리 :contentReference[oaicite:4]{index=4}
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        iou, dice, _ = iou_score(out, y)
        avg['loss'].update(loss.item(), x.size(0))
        avg['iou'].update(iou, x.size(0))

        pbar.set_postfix(OrderedDict(loss=avg['loss'].avg, iou=avg['iou'].avg))
        pbar.update(1)
    pbar.close()
    return OrderedDict(loss=avg['loss'].avg, iou=avg['iou'].avg)


@torch.no_grad()
def validate_one_epoch(cfg, loader, model, criterion):
    avg = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice': AverageMeter()}
    model.eval()
    pbar = tqdm(total=len(loader))
    for x, y, _ in loader:
        x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        iou, dice, _ = iou_score(out, y)
        avg['loss'].update(loss.item(), x.size(0))
        avg['iou'].update(iou, x.size(0))
        avg['dice'].update(dice, x.size(0))
        pbar.set_postfix(OrderedDict(loss=avg['loss'].avg, iou=avg['iou'].avg, dice=avg['dice'].avg))
        pbar.update(1)
    pbar.close()
    return OrderedDict(loss=avg['loss'].avg, iou=avg['iou'].avg, dice=avg['dice'].avg)


def main():
    seed_all()
    cfg_ns = parse_args()
    cfg = vars(cfg_ns)

    # 실험 폴더
    if cfg['name'] is None:
        cfg['name'] = f"{cfg['dataset']}_{cfg['arch']}_{'wDS' if cfg['deep_supervision'] else 'woDS'}"
    save_dir = os.path.join(cfg['output_dir'], cfg['name'])
    os.makedirs(save_dir, exist_ok=True)

    # 로그/설정 저장
    with open(os.path.join(save_dir, 'config.yml'), 'w') as f:
        yaml.dump(cfg, f)
    tb = SummaryWriter(save_dir)

    # 모델/손실/옵티마/스케줄러
    model = archs.__dict__[cfg['arch']](
        cfg['num_classes'], cfg['input_channels'], cfg['deep_supervision'],
        embed_dims=cfg['input_list'], no_kan=cfg['no_kan']
    ).cuda()
    criterion = build_criterion(cfg['loss'])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # 데이터
    train_loader, val_loader = make_dataloaders(cfg)

    # 코드 백업(선택)
    for fname in ['train.py','archs.py']:
        if os.path.exists(fname):
            shutil.copy2(fname, save_dir)

    log = OrderedDict(epoch=[], lr=[], loss=[], iou=[], val_loss=[], val_iou=[], val_dice=[])
    best_iou, best_dice, trigger = 0.0, 0.0, 0

    for epoch in range(cfg['epochs']):
        print(f"Epoch [{epoch}/{cfg['epochs']}]")

        tr = train_one_epoch(cfg, train_loader, model, criterion, optimizer)
        va = validate_one_epoch(cfg, val_loader, model, criterion)

        if scheduler is not None:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va['loss'])
            else:
                scheduler.step()

        # 로그
        current_lrs = [pg['lr'] for pg in optimizer.param_groups]
        log['epoch'].append(epoch); log['lr'].append(current_lrs)
        log['loss'].append(tr['loss']); log['iou'].append(tr['iou'])
        log['val_loss'].append(va['loss']); log['val_iou'].append(va['iou']); log['val_dice'].append(va['dice'])
        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log.csv'), index=False)

        tb.add_scalar('train/loss', tr['loss'], epoch)
        tb.add_scalar('train/iou',  tr['iou'],  epoch)
        tb.add_scalar('val/loss',   va['loss'], epoch)
        tb.add_scalar('val/iou',    va['iou'],  epoch)
        tb.add_scalar('val/dice',   va['dice'], epoch)
        tb.add_scalar('val/best_iou_value',  best_iou, epoch)
        tb.add_scalar('val/best_dice_value', best_dice, epoch)

        # 베스트 저장
        trigger += 1
        if va['iou'] > best_iou:
            best_iou, best_dice, trigger = va['iou'], va['dice'], 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            print("=> saved best model | IoU=%.4f | Dice=%.4f" % (best_iou, best_dice))

        # early stopping
        if cfg['early_stopping'] >= 0 and trigger >= cfg['early_stopping']:
            print("=> early stopping")
            break


if __name__ == '__main__':
    main()

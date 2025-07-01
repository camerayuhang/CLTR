import time
import torch
import numpy as np
from torch import nn

from Networks.CDETR import build_model

# from models.GPNet import GPNet
# from models.LFCNet import LFCNet
# from models.SonarCountNet.DA_ResNet_Transformer_SFABE_CCSI_CAF import Baseline
# from models.CCTrans import alt_gvt_small as cctrans
# from models.marnet import MARNet
# from models.M_SFANet import Model as M_SFANet
# from models.SFANet import Model as SFANet
# from models.SCAR import SCAR
from config import return_args, args

import argparse

return_args = argparse.Namespace(
    dataset='aris',
    save_path='save_file/A_ddp',
    workers=2,
    print_freq=200,
    start_epoch=0,
    epochs=1500,
    pre='None',
    batch_size=8,
    crop_size=320,
    lr_step=1200,
    seed=1,
    best_pred=100000.0,
    gpu_id='0',
    lr=0.0001,
    weight_decay=0.0005,
    save=True,
    scale_aug=True,
    scale_type=1,
    scale_p=0.3,
    gray_aug=True,
    gray_p=0.1,
    test_patch=True,
    channel_point=3,
    num_patch=1,
    min_num=-1,
    num_knn=4,
    test_per_epoch=5,
    threshold=0.35,
    video_path='./video_demo/1.mp4',
    local_rank=-1,
    lr_backbone=0.0001,
    lr_drop=40,
    clip_max_norm=0.1,
    frozen_weights=None,
    backbone='resnet50',
    dilation=False,
    position_embedding='sine',
    enc_layers=6,
    dec_layers=6,
    dim_feedforward=2048,
    hidden_dim=256,
    dropout=0.1,
    nheads=8,
    num_queries=500,
    pre_norm=False,
    masks=False,
    aux_loss=True,
    set_cost_class=2,
    set_cost_point=5,
    set_cost_giou=2,
    mask_loss_coef=1,
    dice_loss_coef=1,
    cls_loss_coef=2,
    count_loss_coef=2,
    point_loss_coef=5,
    giou_loss_coef=2,
    focal_alpha=0.25,
    dataset_file='crowd_data',
    coco_path=None,
    coco_panoptic_path=None,
    remove_difficult=False,
    output_dir='',
    device='cuda',
    resume='',
    eval=False,
    num_workers=2,
    world_size=1,
    dist_url='env:// ',
    master_port=29501,
    distributed=False
)


# 设置设备
device = 'cuda'
# 加载模型
# return_args = Namespace(dataset='aris', save_path='save_file/A_ddp', workers=2, print_freq=200, start_epoch=0, epochs=1500, pre='None', batch_size=8, crop_size=320, lr_step=1200, seed=1, best_pred=100000.0, gpu_id='0', lr=0.0001, weight_decay=0.0005, save=True, scale_aug=True, scale_type=1, scale_p=0.3, gray_aug=True, gray_p=0.1, test_patch=True, channel_point=3, num_patch=1, min_num=-1, num_knn=4, test_per_epoch=5, threshold=0.35, video_path='./video_demo/1.mp4', local_rank=-1, lr_backbone=0.0001, lr_drop=40, clip_max_norm=0.1, frozen_weights=None, backbone='resnet50', dilation=False, position_embedding='sine', enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=256, dropout=0.1, nheads=8, num_queries=500, pre_norm=False, masks=False, aux_loss=True, set_cost_class=2, set_cost_point=5, set_cost_giou=2, mask_loss_coef=1, dice_loss_coef=1, cls_loss_coef=2, count_loss_coef=2, point_loss_coef=5, giou_loss_coef=2, focal_alpha=0.25, dataset_file='crowd_data', coco_path=None, coco_panoptic_path=None, remove_difficult=False, output_dir='', device='cuda', resume='', eval=False, num_workers=2, world_size=1, dist_url='env:// ', master_port=29501, distributed=False)
model_name = "CLTR"
model, criterion, postprocessors = build_model(return_args)
net = nn.DataParallel(model, device_ids=[0])
# model_name = "SCAR"
# net = SCAR(load_weights=True).to(device)  # SCAR
# model_name = "SFANet"
# net = SFANet().to(device)  # SFANet
# model_name = "M-SFANet"
# net = M_SFANet().to(device)  # M_SFANet
# model_name = "MARUNet"
# net = MARNet(objective="dmp+amp").to(device)  # MARNet
# model_name = "CCTrans"
# net = cctrans().to(device)  # CCTrans
# net = LFCNet().to(device)  # LFCNet
# model_name = "GPNet"
# net = GPNet().to(device)  # GPNet
# model_name = "SonarCountNet"
# net = Baseline(pretrained=False).to(device)  # SonarCountNet

net.eval()

# 随机初始化输入数据
x = torch.rand((1, 3, 576, 320)).to(device)
t_all = []


for i in range(100):
    t1 = time.time()
    y = net(x)
    t2 = time.time()

    # 跳过前 5 次的计时
    if i >= 5:
        t_all.append(t2 - t1)

# 统计时间和 FPS
average_time = np.mean(t_all)
average_fps = 1 / average_time
fastest_time = min(t_all)
fastest_fps = 1 / fastest_time
slowest_time = max(t_all)
slowest_fps = 1 / slowest_time
total_time = sum(t_all)

# 打印结果
print(f"Model name: {model_name}")
print(f"Total inference time: {total_time:.6f} seconds")
print(f"Average time per image: {average_time:.6f} seconds")
print(f"Average FPS: {average_fps:.2f}")
print(f"Fastest time: {fastest_time:.6f} seconds (Index: {t_all.index(fastest_time)})")
print(f"Fastest FPS: {fastest_fps:.2f}")
print(f"Slowest time: {slowest_time:.6f} seconds (Index: {t_all.index(slowest_time)})")
print(f"Slowest FPS: {slowest_fps:.2f}")

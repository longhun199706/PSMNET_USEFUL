import time
import argparse
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.PSMnet import PSMNet
from models.smoothloss import SmoothL1Loss
from dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad
import tensorboardX as tX
import matplotlib
import matplotlib.pyplot as plt
from models.PSMnet import PSMNet

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
parser.add_argument('--logdir', default='log/runs', help='log directory')
parser.add_argument('--datadir', default='/root/data/Share data/data_scene_flow/data_scene_flow', help='data directory')
parser.add_argument('--cuda', type=int, default=0, help='gpu number')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--validate-batch-size', type=int, default=1, help='batch size')
parser.add_argument('--log-per-step', type=int, default=1, help='log per step')
parser.add_argument('--save-per-epoch', type=int, default=1, help='save model per epoch')
parser.add_argument('--model-dir', default='checkpoint', help='directory where save model checkpoint')
parser.add_argument('--model-path', default='/root/data/libohan/libohan2/checkpoint/chushi.ckpt', help='path of model to load')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num-epochs', type=int, default=2, help='number of training epochs')
parser.add_argument('--num-workers', type=int, default=8, help='num workers in loading data')
args = parser.parse_args()


mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0, 1]
writer = tX.SummaryWriter(log_dir=args.logdir, comment='PSMNet')
device = torch.device('cuda')
print(device)


def main(args):
    train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
    train_dataset = KITTI2015(args.datadir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    validate_dataset = KITTI2015(args.datadir, mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=args.validate_batch_size, num_workers=args.num_workers)
    step = 0
    best_error = 100.0
    model = PSMNet(args.maxdisp).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)  # 模型并行运行
    criterion = SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        best_error = state['error']
        print('load model from {}'.format(args.model_path))
    else:
        print('66666')
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    for epoch in range(1, args.num_epochs + 1):
        time_start = time.time()
        model.eval()
        error = validate(model, validate_loader, epoch)
        # best_error = save(model, optimizer, epoch, step, error, best_error)
        time_end = time.time()
        print('该epoch运行时间：', time_end - time_start, '秒')


def validate(model, validate_loader, epoch):
    # validate 40 image pairs
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)
    avg_error = 0.0
    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)
        mask = (target_disp > 0)
        mask = mask.detach_()
        with torch.no_grad():
            _, _, disp = model(left_img, right_img)
        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100
        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp
            #save_image(left_save[0], disp_save[0], epoch)
    avg_error = avg_error / num_batches
    print('epoch: {:03} | 3px-error: {:.5}%'.format(epoch, avg_error))
    writer.add_scalar('error/3px', avg_error, epoch)
    # save_image(left_save[0], disp_save[0], epoch)
    return avg_error


if __name__ == '__main__':
    main(args)
    writer.close()

'''
torch.manual_seed(2.0)
model = PSMNet(16).cuda()
left = torch.randn(2, 3, 256, 256).cuda()
right = torch.randn(2, 3, 256, 256).cuda()
print(left[:, :, 0, 0])

out1, out2, out3 = model(left, right)
print(out2[0, :3, :3])
print('66666666777777')
'''
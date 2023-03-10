import os
import argparse
from model import ORNet, GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from util import npmat2euler
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from rich.progress import track

torch.backends.cudnn.enabled = False  # fix cudnn non-contiguous error


def test_one_epoch(args, net, test_loader):
    with torch.no_grad():
        net.eval()

        R_list = []
        t_list = []
        R_pred_list = []
        t_pred_list = []
        euler_list = []

        for src, target, R, t, euler in track(test_loader):
            src = src.cuda()
            target = target.cuda()
            R = R.cuda()
            t = t.cuda()

            R_pred, t_pred, *_ = net(src, target)

            R_list.append(R.detach().cpu().numpy())
            t_list.append(t.detach().cpu().numpy())
            R_pred_list.append(R_pred.detach().cpu().numpy())
            t_pred_list.append(t_pred.detach().cpu().numpy())
            euler_list.append(euler.numpy())

        R = np.concatenate(R_list, axis=0)
        t = np.concatenate(t_list, axis=0)
        R_pred = np.concatenate(R_pred_list, axis=0)
        t_pred = np.concatenate(t_pred_list, axis=0)
        euler = np.concatenate(euler_list, axis=0)

        euler_pred = npmat2euler(R_pred)
        r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
        r_rmse = np.sqrt(r_mse)
        r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
        t_mse = np.mean((t - t_pred) ** 2)
        t_rmse = np.sqrt(t_mse)
        t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae


def train_one_epoch(args, net, train_loader, opt):
    net.train()

    R_list = []
    t_list = []
    R_pred_list = []
    t_pred_list = []
    euler_list = []


    for src, target, R, t, euler in track(train_loader):
        src = src.cuda()
        target = target.cuda()
        R = R.cuda()
        t = t.cuda()

        opt.zero_grad()
        R_pred, t_pred, loss = net(src, target, R, t)

        R_list.append(R.detach().cpu().numpy())
        t_list.append(t.detach().cpu().numpy())
        R_pred_list.append(R_pred.detach().cpu().numpy())
        t_pred_list.append(t_pred.detach().cpu().numpy())
        euler_list.append(euler.numpy())

        loss.backward()
        opt.step()

    R = np.concatenate(R_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    R_pred = np.concatenate(R_pred_list, axis=0)
    t_pred = np.concatenate(t_pred_list, axis=0)
    euler = np.concatenate(euler_list, axis=0)


    euler_pred = npmat2euler(R_pred)
    r_mse = np.mean((euler_pred - np.degrees(euler)) ** 2)
    r_rmse = np.sqrt(r_mse)
    r_mae = np.mean(np.abs(euler_pred - np.degrees(euler)))
    t_mse = np.mean((t - t_pred) ** 2)
    t_rmse = np.sqrt(t_mse)
    t_mae = np.mean(np.abs(t - t_pred))

    return r_rmse, r_mae, t_rmse, t_mae


def train(args, net, train_loader, test_loader):
    opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = MultiStepLR(opt, milestones=[30, 60, 80], gamma=0.1)

    info_test_best_rot_RMSE = None
    for epoch in range(args.epochs):

        train_stats = train_one_epoch(args, net, train_loader, opt)
        test_stats = test_one_epoch(args, net, test_loader)

        print('=====  EPOCH %d  =====' % (epoch + 1))
        print('TRAIN, rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % train_stats)
        print('TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % test_stats)
        rot_RMSE, _, _, _ = test_stats
        if epoch>0 and epoch % 5 == 0:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        if info_test_best_rot_RMSE is None or rot_RMSE < info_test_best_rot_RMSE:
            info_test_best_rot_RMSE = rot_RMSE
            torch.save(net.state_dict(), 'checkpoints/%s/models/model_c_best.t7' % (args.exp_name))
        scheduler.step()

    print('Finished!')


def test(args, net, test_loader):
    net.load_state_dict(torch.load('checkpoints/exp-unseen/model_c_best.t7'))

    test_stats = test_one_epoch(args, net, test_loader)
    print('TEST,  rot_RMSE: %f, rot_MAE: %f, trans_RMSE: %f, trans_MAE: %f' % test_stats)


def main():
    arg_bool = lambda x: x.lower() in ['true', 't', '1']
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp-unseen', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--num_iter', type=int, default=3, metavar='N',
                        help='Number of iteration inside the network')
    parser.add_argument('--emb_dims', type=int, default=64, metavar='N',
                        help='Dimension of embeddings.')
    parser.add_argument('--batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=6, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--unseen', type=arg_bool, default='False',
                        help='Test on unseen categories')
    parser.add_argument('--gaussian_noise', type=arg_bool, default='False',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--alpha', type=float, default=0.75, metavar='N',
                        help='Fraction of points when sampling partial point cloud')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')

    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=128, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')

    args = parser.parse_args()
    print(args)

    ##### make checkpoint directory and backup #####
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    ##### make checkpoint directory and backup #####

    ##### load data #####
    train_loader = DataLoader(
        ModelNet40(partition='train', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen,
                   factor=args.factor),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(
        ModelNet40(partition='test', alpha=args.alpha, gaussian_noise=args.gaussian_noise, unseen=args.unseen,
                   factor=args.factor),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    ##### load data #####

    ##### load model #####

    net = ORNet(args).cuda()
    ##### load model #####

    ##### train #####
    train(args, net, train_loader, test_loader)
    ##### train #####
    # test(args, net, test_loader)

if __name__ == '__main__':
    main()

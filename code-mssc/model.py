import os
import sys
import glob
import h5py
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import knn, batch_choice

from transformer import Transformer


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i + 1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Propagate(nn.Module):
    def __init__(self, in_channel, emb_dims):
        super(Propagate, self).__init__()
        self.conv2d = Conv2DBlock((in_channel, emb_dims, emb_dims), 1)
        self.conv1d = Conv1DBlock((emb_dims, emb_dims), 1)

    def forward(self, x, idx):
        batch_idx = np.arange(x.size(0)).reshape(x.size(0), 1, 1)
        nn_feat = x[batch_idx, :, idx].permute(0, 3, 1, 2)
        x = nn_feat - x.unsqueeze(-1)
        x = self.conv2d(x)
        x = x.max(-1)[0]
        x = self.conv1d(x)
        return x


class GNN(nn.Module):
    def __init__(self, emb_dims=64):
        super(GNN, self).__init__()
        self.propogate1 = Propagate(3, 64)
        self.propogate2 = Propagate(64, 64)
        self.propogate3 = Propagate(64, 64)
        self.propogate4 = Propagate(64, 64)
        self.propogate5 = Propagate(64, emb_dims)

    def forward(self, x):
        nn_idx = knn(x, k=12)

        x = self.propogate1(x, nn_idx)
        x = self.propogate2(x, nn_idx)
        x = self.propogate3(x, nn_idx)
        x = self.propogate4(x, nn_idx)
        x = self.propogate5(x, nn_idx)

        return x


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src, src_corr, weights):
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + (
                    weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)
        return R, t.view(src.size(0), 3)


def get_mask(src_gt, tgt):
    dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
    min_dist_st, src_min_idx = (dist ** 2).sum(1).min(-1)
    min_dist_ts, tgt_min_idx = (dist ** 2).sum(1).min(-2)
    min_dist_st, min_dist_ts = torch.sqrt(min_dist_st), torch.sqrt(min_dist_ts)
    s_mask, t_mask = (min_dist_st < 0.05).float(), (min_dist_ts < 0.05).float()

    return s_mask, t_mask


def get_relative_distance(points):
    center = points.mean(-1, keepdim=True)
    dist = points.unsqueeze(-1) - center.unsqueeze(-2)
    relative_dist = torch.sqrt((dist ** 2).sum(1))
    return relative_dist


def mask_change(mask, tk):
    # mask[B,N],tk[B,k],means-topk
    B, N = mask.shape
    res = torch.zeros_like(mask)
    for i in range(B):
        res[i] = res[i].index_fill(-1, tk[i], 1.0)

    return res


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.emb_dims = emb_dims
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, emb_dims, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        B, _, n_pts = x.size()
        x = F.relu(self.bn4(self.conv4(x)))
        pointfeat2 = x
        x = F.relu(self.bn5(self.conv5(x)))
        pointfeat3 = x
        x = torch.max(x, 2, keepdim=True)[0]
        globalfeat = x.view(-1, self.emb_dims, 1).repeat(1, 1, n_pts)

        return torch.cat([pointfeat2, pointfeat3, globalfeat], 1), globalfeat
        # return torch.cat([pointfeat3,globalfeat],1),globalfeat


class Upnet(nn.Module):
    def __init__(self, emb_dims=64):
        super(Upnet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, emb_dims, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        B, _, n_pts = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ORNet(nn.Module):
    def __init__(self, args):
        super(ORNet, self).__init__()

        self.ahead_nn = PointNet()
        self.num_iter = args.num_iter
        self.emb_nn = GNN()
        self.pointer = Transformer(64, 1, 0.0, 128, 4)
        self.pointer1 = Transformer(64, 1, 0.0, 128, 4)

        self.upnet = Upnet()
        self.pointer_up = Transformer(64, 1, 0.0, 128, 4)

        self.significance = Conv1DBlock((1536 + 128, 256, 128, 64, 1), 1)

        self.head = SVDHead(args=args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, tgt, R_gt=None, t_gt=None):

        if not (self.training or (R_gt is None and t_gt is None)):
            raise Exception('Passing ground truth while testing')

        if self.training:
            src_gt = torch.matmul(R_gt, src) + t_gt.unsqueeze(-1)
            s_mask, t_mask = get_mask(src_gt, tgt)
            R_ba = torch.inverse(R_gt)
            t_ba = -torch.matmul(R_ba, t_gt.unsqueeze(-1)).squeeze(-1)

        R = torch.eye(3).unsqueeze(0).expand(src.size(0), -1, -1).cuda().float()
        t = torch.zeros(src.size(0), 3).cuda().float()
        loss = 0.
        batch_size, _, num_points = src.size()
        src_copy = copy.deepcopy(src)
        tgt_copy = copy.deepcopy(tgt)


        for i in range(self.num_iter):

            src_up_emb = self.upnet(src)
            tgt_up_emb = self.upnet(tgt)

            src_up, tgt_up = self.pointer_up(src_up_emb, tgt_up_emb)
            src_up_emb = src_up_emb + src_up
            tgt_up_emb = tgt_up_emb + tgt_up

            src_ahead_emb, src_ahead_global = self.ahead_nn(src_up_emb)
            tgt_ahead_emb, tgt_ahead_global = self.ahead_nn(tgt_up_emb)

            src_in = torch.cat([src_ahead_emb, tgt_ahead_global], dim=1)
            tgt_in = torch.cat([tgt_ahead_emb, src_ahead_global], dim=1)

            s_sig_score = self.sigmoid(self.significance(src_in).squeeze(1))
            t_sig_score = self.sigmoid(self.significance(tgt_in).squeeze(1))

            if self.training:
                loss_mask = nn.BCELoss()(s_sig_score, s_mask) + nn.BCELoss()(t_sig_score, t_mask)

                loss = loss + loss_mask

            tgt_embedding = self.emb_nn(tgt)
            src_embedding = self.emb_nn(src)
            #
            src_embedding_p, _ = self.pointer(src_embedding, src_embedding)
            tgt_embedding_p, _ = self.pointer(tgt_embedding, tgt_embedding)
            src_embedding = src_embedding + src_embedding_p
            tgt_embedding = tgt_embedding + tgt_embedding_p

            src_embedding_p2, tgt_embedding_p2 = self.pointer1(src_embedding, tgt_embedding)

            src_embedding = src_embedding + src_embedding_p2
            tgt_embedding = tgt_embedding + tgt_embedding_p2

            num_point_preserved = num_points // 2

            if self.training:
                src_idx_candidate = s_mask.topk(k=num_point_preserved, dim=-1)[1]
                src_pred = mask_change(s_mask, src_idx_candidate)
                indicator = src_pred.cpu().numpy()
                indicator += 1e-5
                probs = indicator / indicator.sum(-1, keepdims=True)
                candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
                src_idx_K = batch_choice(candidates, num_points // 6, p=probs)

                tgt_idx_candidate = t_mask.topk(k=num_point_preserved, dim=-1)[1]
                tgt_pred = mask_change(t_mask, tgt_idx_candidate)
                indicators = tgt_pred.cpu().numpy()
                indicators += 1e-5
                probss = indicators / indicators.sum(-1, keepdims=True)
                candidatess = np.tile(np.arange(tgt.size(-1)), (tgt.size(0), 1))
                tgt_idx_K = batch_choice(candidatess, num_points // 6, p=probss)

            else:
                src_idx_K = s_sig_score.topk(k=num_points // 2, dim=-1)[1]
                src_idx_K = src_idx_K.cpu().numpy()

                tgt_idx_K = t_sig_score.topk(k=num_points // 2, dim=-1)[1]
                tgt_idx_K = tgt_idx_K.cpu().numpy()

            batch_idx = np.arange(src.size(0))[:, np.newaxis]

            src_K = src[batch_idx, :, src_idx_K].transpose(1, 2)
            src_K1 = src_copy[batch_idx, :, src_idx_K].transpose(1, 2)
            src_Ks_embedding = src_embedding[batch_idx, :, src_idx_K].transpose(1, 2)

            tgt_K = tgt[batch_idx, :, tgt_idx_K].transpose(1, 2)
            tgt_K1 = tgt_copy[batch_idx, :, tgt_idx_K].transpose(1, 2)
            tgt_Ks_embedding = tgt_embedding[batch_idx, :, tgt_idx_K].transpose(1, 2)

            d_k = src_Ks_embedding.size(1)

            matrix_sK_t = torch.matmul(src_Ks_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)

            matrix_tK_s = torch.matmul(tgt_Ks_embedding.transpose(2, 1).contiguous(), src_embedding) / math.sqrt(d_k)

            scores_st = torch.softmax(matrix_sK_t, dim=2)

            scores_ts = torch.softmax(matrix_tK_s, dim=2)

            srcK_corr = torch.matmul(tgt, scores_st.transpose(2, 1).contiguous())

            tgtK_corr = torch.matmul(src, scores_ts.transpose(2, 1).contiguous())

            weights = scores_st.max(-1)[0]
            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

            weights1 = scores_ts.max(-1)[0]
            weights1 = torch.sigmoid(weights1)
            weights1 = weights1 * (weights1 >= weights1.median(-1, keepdim=True)[0]).float()
            weights1 = weights1 / (weights1.sum(-1, keepdim=True) + 1e-8)

            if self.training:
                loss_st = F.mse_loss((torch.matmul(R_gt, src_K1) + t_gt.unsqueeze(-1)), srcK_corr)

                loss_ts = F.mse_loss(
                    (torch.matmul(R, torch.matmul(R_ba, tgt_K1) + t_ba.unsqueeze(-1)) + t.unsqueeze(-1)), tgtK_corr)
                loss = loss + loss_st + loss_ts


            src_K = torch.cat([src_K, tgtK_corr], dim=-1)
            srcK_corr = torch.cat([srcK_corr, tgt_K], dim=-1)
            weights = torch.cat([weights, weights1], dim=-1)
            weights = torch.sigmoid(weights)
            weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

            rotation_st, translation_st = self.head(src_K, srcK_corr, weights)

            rotation_st = rotation_st.detach()
            translation_st = translation_st.detach()
            rotation_ts = rotation_ts.detach()
            translation_ts = translation_ts.detach()

            src = torch.matmul(rotation_st, src) + translation_st.unsqueeze(-1)
            R = torch.matmul(rotation_st, R)
            t = torch.matmul(rotation_st, t.unsqueeze(-1)).squeeze() + translation_st

        return R, t, loss



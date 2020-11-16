import torch
import torch.nn as nn
from .Histogram import *

class styleLoss(nn.Module):
    def forward(self, input, target):
        ib, ic, ih, iw = input.size()
        iF = input.view(ib, ic, -1)
        iMean = torch.mean(iF, dim=2)
        iCov = GramMatrix()(input)

        tb,tc,th,tw = target.size()
        tF = target.view(tb, tc, -1)
        tMean = torch.mean(tF, dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(size_average=False)(iMean, tMean) + nn.MSELoss(size_average=False)(iCov, tCov)
        return loss / tb

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div_(c * h * w)

######################### ADDED #########################
class histogramLoss(nn.Module):
    def compress(self, f, range, gamma):
        f = (f - range[0]) / (range[1] - range[0])
        return torch.pow(f, (1 / gamma))

    def decompress(self, f, range, gamma):
        f = torch.pow(f, gamma)
        f = f * (range[1] - range[0]) + range[0]
        return f

    def featurewiseHistogramMatch(self, iF, tF, M, N, bins):
        iF = torch.squeeze(iF, 0)
        iF = iF.view(N, -1)
        tF = torch.squeeze(tF, 0)
        tF = tF.view(N, -1)

        tA = torch.empty(N)

        for i in range(N):
            tF_to_iF = matchHistogram(iF[i], tF[i], bins)
            tA.scatter_add(i, tF_to_iF)

        return tA, tF

    def forward(self, input, target, bins, gamma=4.0):
        ib, ic, ih, iw = input.size()
        iF = input
        # iF = input.view(ib, ic, -1)

        tb, tc, th, tw = target.size()
        tF = target
        # tF = target.view(tb, tc, -1)

        range = [torch.min(iF), torch.max(iF)]
        iF = self.compress(iF, range, gamma)
        tF = self.compress(tF, range, gamma)

        # tF_to_iF, tF = self.featurewiseHistogramMatch(iF, tF, ih * iw, ic, bins)
        # tF_to_iF = tF_to_iF.detach()
        tF_to_iF = tF.detach()
        tF = self.decompress(tF_to_iF, range, gamma)

        loss = (1. / (ih * iw * ic)) * torch.sum(torch.pow((tF - tF_to_iF), 2))
        return loss

class tvLoss(nn.Module):
    def forward(self, input):
        ib, ic, ih, iw = input.size()

        tv_loss_y = nn.MSELoss()(input[:, 1:, :, :], input[:, :-1, :, :])
        tv_loss_y /= (ib * (ih - 1) * iw * ic)
        tv_loss_x = nn.MSELoss()(input[:, :, 1:, :], input[:, :, :-1, :])
        tv_loss_x /= (ib * ih * (iw - 1) * ic)

        tv_loss = tv_loss_y + tv_loss_x
        return tv_loss / ib

######################### ADDED #########################

######################## MODIFIED #######################
class LossCriterion(nn.Module):
    def __init__(self, style_layers, content_layers, histogram_layers, style_weight, content_weight, histogram_weight, histogram_bins, tv_layers, tv_weight):
        super(LossCriterion,self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.histogram_layers = histogram_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.histogram_weight = histogram_weight
        self.histogram_bins = histogram_bins
        self.tv_layers = tv_layers
        self.tv_weight = tv_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.histogramLosses = [histogramLoss()] * len(histogram_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)
        self.tvLosses = [tvLoss()] * len(tv_layers)

    def forward(self, tF, sF, cF):
        # content loss
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            cf_i = cF[layer]
            cf_i = cf_i.detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # style loss
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer]
            sf_i = sf_i.detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i, sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight

######################### ADDED #########################
        totalHistogramLoss = 0
        for i, layer in enumerate(self.histogram_layers):
            sf_i = sF[layer]
            tf_i = tF[layer]
            loss_i = self.histogramLosses[i]
            totalHistogramLoss += loss_i(tf_i, sf_i, self.histogram_bins)
        totalHistogramLoss = totalHistogramLoss * self.histogram_weight

        totalTvLoss = 0
        for i, layer in enumerate(self.tv_layers):
            tf_i = tF[layer]
            loss_i = self.tvLosses[i]
            totalTvLoss += loss_i(tf_i)
        totalTvLoss = totalTvLoss * self.tv_weight

######################### ADDED #########################

        loss = totalStyleLoss + totalContentLoss + totalHistogramLoss + totalTvLoss
        return loss, totalStyleLoss, totalContentLoss, totalHistogramLoss, totalTvLoss
######################## MODIFIED #######################

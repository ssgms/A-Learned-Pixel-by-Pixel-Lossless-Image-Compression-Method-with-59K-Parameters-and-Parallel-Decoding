import torch
from torch import nn
from torch.nn import functional as F
import torch.distributions as tdist
from graphs.layers.masked_conv2d import MaskedConv2d
from graphs.layers.res_nets import ResBlock2d
from compressai.ops import LowerBound

class CondGMMNetMaskedConvMethod3(nn.Module):
    def __init__(self, config, num_mix, num_mdl=3, res_con=False):
        super(CondGMMNetMaskedConvMethod3, self).__init__()
        scale_bound: float = 0.11/255.0
        self.lower_bound_scale = LowerBound(scale_bound)
        self.cnv_chn = 128
        self.device = config.device
        self.num_mix = num_mix
        self.num_mdl = num_mdl
        self.res_con = res_con
        self.out_chn = 3 * self.num_mix * self.num_mdl + self.num_mix * int((self.num_mdl - 1) * self.num_mdl / 2.0)
        self.delta = 0.001# delta to be used in derivative approximation of cdf: pdf(x)=(cdf(x+delta)-cdf(x-delta))/(2*delta)
        self.bound = nn.Parameter(1e-9 * torch.ones(1), requires_grad=False)# small bound, so that cdf difference (i.e. probability) is not smaller than this; otherwise gradient problems and nan values can occur in training
        self.normal = tdist.Normal(0, 1)
        if not self.res_con:
            self.get_wms = nn.Sequential( # generates means, stdevs, weights for GMM-based entropy model using masked conv
                MaskedConv2d('A', self.num_mdl,  self.cnv_chn, 5, 1, 5//2, bias=True, padding_mode='zeros'), # Don't replicate!!
                nn.LeakyReLU(),
                MaskedConv2d('B',  self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                nn.LeakyReLU(),
                MaskedConv2d('B', self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                nn.LeakyReLU(),
                MaskedConv2d('B', self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                nn.LeakyReLU(),
                MaskedConv2d('B', self.cnv_chn, self.out_chn, 1, 1, 1//2, bias=True)
            )
        else:
            self.get_wms = nn.Sequential( # generates means, stdevs, weights for GMM-based entropy model using masked conv
                MaskedConv2d('A', self.num_mdl, self.cnv_chn, 5, 1, 5//2, bias=True, padding_mode='zeros'), # Don't replicate!!
                ResBlock2d('B', self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                ResBlock2d('B', self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                ResBlock2d('B', self.cnv_chn, self.cnv_chn, 1, 1, 1//2, bias=True),
                nn.Conv2d(self.cnv_chn, self.out_chn, 1)
            )
    def forward_wms(self, org):
        wms = self.get_wms(org)
        return wms

    def forward_params(self,org,wms):
        P = self.num_mdl  # RGB size, i.e. 3 pixels for RGB or 1 pixel for gray-scale (equal to C of x)
        M = self.num_mix  # num of mixtures in GMM model
        means = torch.zeros([wms.shape[0], P * M, wms.shape[2], wms.shape[3]], device=self.device)

        means[:, :, :, :] = wms[:, 0 * P * M:1 * P * M, :, :]  # B x PM x H x W
        sdevs = wms[:, 1 * P * M:2 * P * M, :, :]  # B x PM x H x W
        weights = wms[:, 2 * P * M:3 * P * M, :, :]  # B x PM x H x W

        means[:, 1 * M:2 * M, :, :] = means[:, 1 * M:2 * M, :, :] \
                                      + wms[:, 9 * M:10 * M, :, :] * org[:, 0, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1)
        # update mean of B pixels based on origR,origG pixels
        means[:, 2 * M:3 * M, :, :] = means[:, 2 * M:3 * M, :, :] \
                                      + wms[:, 10 * M:11 * M, :, :] * org[:, 0, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1) \
                                      + wms[:, 11 * M:12 * M, :, :] * org[:, 1, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1)
        scales = self.lower_bound_scale(sdevs)
        return means,weights,scales

    def forward_cdf(self, x, means,weights,scales):
        B=x.shape[0]#batch size
        P = self.num_mdl  # RGB size, i.e. 3 pixels for RGB or 1 pixel for gray-scale (equal to C of x)
        M = self.num_mix  # num of mixtures in GMM model
        ccc = self.normal.cdf( (x.repeat_interleave(M, dim=1) - means) /scales) # B x PM x H x W
        cdfs = torch.sum(ccc.view(B, P, M, x.shape[2], x.shape[3]) * F.softmax(weights.view(B, P, M, x.shape[2], x.shape[3]), dim=2), dim=2)
        return cdfs

    def forward_pmf(self, x):
        """
        NOTE : this is the function that needs to be called to train our compression model
        :param x: input images/patches  # B x C x H x W
        :return: PMF values (probability within [sample-0.5, sample+0.5]) ,   # B x C x H x W
        """
        wms = self.forward_wms(x)
        means, weights, scales=self.forward_params(x, wms)
        pmf_int = self.forward_cdf(x + 0.5 / 255, means, weights, scales) - self.forward_cdf(x - 0.5 / 255, means, weights, scales)
        pmf_int_clip = torch.max(pmf_int, self.bound)
        return pmf_int_clip

    ############################################# FOR REAL COMPRESSION ########################################
    def forward_cdf_encode(self,x,org,wms):
        B = x.shape[0]  # batch size
        P = self.num_mdl  # RGB size, i.e. 3 pixels for RGB or 1 pixel for gray-scale (equal to C of x)
        M = self.num_mix  # num of mixtures in GMM model
        means=torch.zeros([wms.shape[0],P*M,wms.shape[2],wms.shape[3]],device=self.device)

        means[:,:,:,:] = wms[:, 0 * P * M:1 * P * M, :, :]  # B x PM x H x W
        sdevs = wms[:, 1 * P * M:2 * P * M, :, :]  # B x PM x H x W
        weigs = wms[:, 2 * P * M:3 * P * M, :, :]  # B x PM x H x W

        means[:, 1*M:2*M, :, :] = means[:, 1*M:2*M, :, :] \
                                  + wms[:, 9*M:10*M, :, :] * org[:,0,:,:].unsqueeze(dim=1).repeat_interleave(M, dim=1) \

        # update mean of B pixels based on origR,origG pixels
        means[:, 2*M:3*M, :, :] = means[:, 2*M:3*M, :, :] \
                                  + wms[:, 10*M:11*M, :, :] * org[:,0,:,:].unsqueeze(dim=1).repeat_interleave(M, dim=1) \
                                  + wms[:, 11*M:12*M, :, :] * org[:,1,:,:].unsqueeze(dim=1).repeat_interleave(M, dim=1) \

        scales = self.lower_bound_scale(sdevs)
        norm_x=(x-means.unsqueeze(-1))/ scales.unsqueeze(-1)
        ccc= self.normal.cdf(norm_x)
        weights=F.softmax(weigs.view(1, P, M, wms.shape[2], wms.shape[3]), dim=2)
        cdfs = torch.sum((ccc).view(1, P, M, wms.shape[2], wms.shape[3],B) *  weights.unsqueeze(-1),dim=2)
        cdfs=cdfs[0].permute(3,0,1,2)
        return cdfs

    def forward_cdf_decode(self, x, org, wms, c):
        P = self.num_mdl  # RGB size, i.e. 3 pixels for RGB or 1 pixel for gray-scale (equal to C of x)
        M = self.num_mix  # num of mixtures in GMM model
        means = torch.zeros([wms.shape[0], P * M, wms.shape[2], wms.shape[3]], device=self.device)
        means[:, :, :, :] = wms[:, 0 * P * M:1 * P * M, :, :]  # B x PM x H x W
        sdevs = wms[:, 1 * P * M:2 * P * M, :, :]  # B x PM x H x W
        weights = wms[:, 2 * P * M:3 * P * M, :, :]  # B x PM x H x W

        if c == 1:
            means[:, 1 * M:2 * M, :, :] = means[:, 1 * M:2 * M, :, :] \
                                          + wms[:, 9 * M:10 * M, :, :] * org[:, 0, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1)
        if c == 2:
            means[:, 2 * M:3 * M, :, :] = means[:, 2 * M:3 * M, :, :] \
                                  + wms[:, 10 * M:11 * M, :, :] * org[:, 0, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1) \
                                  + wms[:, 11 * M:12 * M, :, :] * org[:, 1, :, :].unsqueeze(dim=1).repeat_interleave(M, dim=1) \

        scales = self.lower_bound_scale(sdevs)
        norm_x = (x - (means[:, c * M:(c + 1) * M, 2, 2]).unsqueeze(-1)) / (
        scales[:, c * M:(c + 1) * M, 2, 2]).unsqueeze(-1)
        ccc = self.normal.cdf(norm_x)
        weights = torch.softmax(weights[:, c * M:(c + 1) * M, :, :], dim=1)
        cdfs = torch.sum(ccc * weights[:, :, 2, 2].unsqueeze(-1), dim=1)
        return cdfs



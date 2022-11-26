import torch
from torch import nn
from ..layers.gmm_mskconv_nets import CondGMMNetMaskedConvMethod3
from ..utils.ae import *
from ..numpyAc.numpyAc import *
import numpy as np
import torch.nn.functional as F


class MaskedConvLosslessNet(nn.Module):
    """ Neural-network architecture that implements Masked Convolution based Lossless Image Compression (?) """
    def __init__(self, config):
        super(MaskedConvLosslessNet, self).__init__()
        self.num_mixtures_GMM = config.num_mix_GMM
        self.num_mdl = config.num_color_channels
        self.entropy = EntropyLayerMaskedConvLS(config)

    def forward(self, y):
        bpp = self.entropy(y) #param y: input images/patches  # B x C x H x W
        return bpp #return: bpp (bits-per-pixel) compression rate # B x C x H x W

    def compress(self, y):
        bpp = self.entropy.lossless_encode(y)
        return bpp

    def decompress(self, h,w):
        bitstream = self.entropy.lossless_decode(h,w)
        return  bitstream

class EntropyLayerMaskedConvLS(nn.Module):
    """ Entropy layer for Masked Convolution based Lossless Image Compression based on GMMs """
    def __init__(self, config):
        """
        Estimates entropy during training; performs quantization/rounding during validation.
        Uses GMM to model conditional PDF(s).
        """
        super(EntropyLayerMaskedConvLS, self).__init__()
        self.num_mix = config.num_mix_GMM
        self.num_mdl = config.num_color_channels
        self.res_con = config.res_con
        self.mode = config.mode
        self.model = CondGMMNetMaskedConvMethod3(config, self.num_mix, self.num_mdl, self.res_con)
        self.run_mode = config.run_mode
        self.wr_img_path = config.write_encode_data
        self.bistream_path = config.bitstream_path
        self.run_type = config.run_type
        self.device = config.device
        self.H = config.H
        self.W = config.W
        self.matrix_file_name1 = config.paralel_data_matrix1
        self.matrix_file_name2 = config.paralel_data_matrix2
        self.decoder_method = config.decoder_method
        self.paralel_data_matrix1_encode = config.paralel_data_matrix1_encode
        self.paralel_data_matrix2_encode = config.paralel_data_matrix2_encode
        self.run_type = config.run_type
        self.codec = arithmeticCoding()

    def forward(self, y):
        """
        See notes if can not easily understand architecture
        :param y: input images/patches  # B x C x H x W
        :return: bpp (bits-per-pixel) compression rate # B x C x H x W
        """
        pmf_values = self.model.forward_pmf(y)  # B x C x H x W
        self_informations = -torch.log2(pmf_values)
        return self_informations

    def lossless_encode(self, y):
        if self.mode=='test':
            if (self.run_type == "S"):
                self_informations = self._arithmetic_code(y)
            elif (self.run_type == "P"):
                self_informations = self._arithmetic_code_parallel(y)
        else:
            pmf_values = self.model.forward_pmf(y)  # B x C x H x W
            self_informations = -torch.log2(pmf_values)
            bpp_rate = torch.mean(self_informations)  # calculating bits per subpixel
            print(bpp_rate, "bpp_rate")
        return self_informations

    def lossless_decode(self,h,w):
        if (self.run_type=="S"):
            decoded_image=self.decode_bitstream(h,w)
        elif (self.run_type=="P"):
            decoded_image = self.decode_bitstream_parallel(h,w)
        return decoded_image
	
	#Some parts of the following code for wavefront parallelization follows Zhang et al's codes at link https://github.com/zmtomorrow/ParallelNeLLoC
    ##################  SERIAL DECODE  #####################
    @torch.no_grad()
    def decode_bitstream(self,h,w,ks=5):
        function = self.model; device=self.device;mid = int(ks / 2)
        decode_img = torch.zeros([1, 3, h + mid * 2, w + mid * 2])
        batch_steps = calculate_batch_steps(self, 3, 5)
        decodec = arithmeticDeCoding(None, h*w*3, 256, self.bistream_path)
        for i in range(0, h):
            for j in range(0, w):
                patch = decode_img[:, :, i:i + mid + 1, j:j + ks] / 255
                wms = function.forward_wms(patch.to(self.device))
                for c in range(0, 3):
                    cdf = function.forward_cdf_decode(batch_steps,patch.to(device), wms,c)
                    cdfs_dec = cdf.cpu().numpy()
                    decoded_pix_np = decodec.decode(cdfs_dec)
                    decode_img[0, c, i + mid, j + mid] = decoded_pix_np
                    patch[0 , c, mid, mid] = decoded_pix_np / 255.
        org_img_array = np.load(self.wr_img_path + 'encoded_img_array.npy')
        print("Lossless  status:",np.array_equiv(org_img_array, decode_img[0, 0:3, mid:h + mid, mid:w + mid]))
        return decode_img[0, 0:3, mid:h + mid, mid:w + mid]

    ################# PARALLEL DECODE #####################
    @torch.no_grad()
    def decode_bitstream_parallel(self,h,w,k=5):
        function = self.model;mid = int(k / 2);
        decode_img = (torch.zeros([1, 3, h + mid * 2, w + mid * 2])).numpy();
        batch_steps = calculate_batch_steps(self, 3, 5)
        decodec = arithmeticDeCoding(None, h * w * 3, 256, self.bistream_path)
        time_index, length = load_paralellize_matrix(self, h, w, k);l = 0
        for par_index_list in time_index:
            par_index_list = list(filter(lambda x: x[0] < h, par_index_list))  # Filter list according to height
            par_index_list = list(filter(lambda x: x[1] < w, par_index_list))  # Filter list according to weight
            patch_list = [];
            if (l == length):break
            l = l + 1
            for i, j in par_index_list:
                patch_list.append(decode_img[0, :, i:i + mid + 1, j:j + k] / 255)
            patches =np.stack(patch_list).reshape(-1,3,3,5)
            if (self.decoder_method == 2):
               patches[:,:,0,4] = patches[:,:,0,3];patches[:,:,1,3] = patches[:,:,0,3];patches[:,:,1,4] = patches[:,:,0,3]
            wms = function.forward_wms(torch.from_numpy(patches).float().to(self.device))
            for c in range(0, 3):
                cdf = function.forward_cdf_decode(batch_steps, torch.from_numpy(patches).float().to(self.device),wms,c)
                cdfs_dec = cdf.cpu().numpy()
                for patch_index, (i, j) in enumerate(par_index_list):
                    cdf_numpyac=cdfs_dec[patch_index:patch_index+1,:]
                    decoded_pix_np = decodec.decode(cdf_numpyac)
                    decode_img[0, c, i + mid, j + mid] = decoded_pix_np
                    patches[patch_index, c, mid, mid] = decoded_pix_np / 255.
        org_img_array = np.load(self.wr_img_path + 'encoded_img_array.npy')
        print("Lossless  status:",np.array_equiv(org_img_array, decode_img[0, 0:3, mid:h + mid, mid:w + mid]))
        return decode_img[0, 0:3, mid:h + mid, mid:w + mid]

  ################## SERIAL ENCODE #####################
    @torch.no_grad()
    def _arithmetic_code(self,org):
        org = org[:, :, 0:0 + self.H, 0:0 + self.W];print(org.shape[2],org.shape[3])
        np.save(self.wr_img_path + 'encoded_img_array', (org * 255).cpu())
        function=self.model;H=org.shape[2];W=org.shape[3];
        cdf_input_tensor=calculate_batch_steps(self,H, W)
        wms=function.forward_wms(org)
        cdf_out=[];
        for i in range (257):
            x=cdf_input_tensor[i:i+1]
            cdf_tensor = function.forward_cdf_encode(x,org,wms)
            cdf_out.append(cdf_tensor.cpu())
        cdfs_enc = (((torch.cat(cdf_out, dim=0)).reshape(257, 3, H * W)).permute(2, 1, 0)).reshape(-1, 257); cdfs_enc = torch.clamp(cdfs_enc, min=0, max=1)
        sym_org =((255 * (org)).type(torch.int16)).cpu()
        sym_org=((sym_org.reshape(-1,H*W)).permute(1,0)).reshape(-1)
        byte_stream, real_bits = self.codec.encode(cdfs_enc.numpy(), sym_org.numpy(),self.bistream_path)
        rate = len(byte_stream) * 8 /torch.numel(sym_org);
        print("****Bitrate*** : ", rate)
        return rate

    ################## PARALLEL ENCODE #####################
    @torch.no_grad()
    def _arithmetic_code_parallel(self, org):
        org=org[:,:,0:0+self.H,0:0+self.W]
        np.save(self.wr_img_path + 'encoded_img_array', (org * 255).cpu())
        function=self.model;H = org.shape[2]; W = org.shape[3];H_org=H;W_org=W; K=5;
        patches = get_patches(self, org, H, W, K)
        val=60000;cdf_total=[];num_symbol=H_org*W_org*3
        if (H*W>val):for_finish_flag = 0;H_new=int(val/W)
        else:for_finish_flag=1;H_new=H
        symbol_reshaped=(patches[:, :, 2:3, 2:3].permute(2, 3, 1, 0)).reshape(1, 3, H, W)
        patches=torch.clamp(patches, min=0, max=1)
        batch_step=W * H_new;H_batch=H_new
        for k in range(0,H*W,batch_step):
            if (k+batch_step>H*W):
                batch_step=H*W-k
                H_batch=int(batch_step/W)
            cdf_out = []
            img_patch=patches[k:k+batch_step, :, :, :]
            wms_patch = function.forward_wms(img_patch)
            cdf_input_tensor = calculate_batch_steps(self, H_batch, W)
            patches_reshaped = (img_patch[:, :, 2:3, 2:3].permute(2, 3, 1, 0)).reshape(1, 3, H_batch, W)
            wms_reshaped = (wms_patch[:, :, 2:3, 2:3].permute(2, 3, 1, 0)).reshape(1, wms_patch.shape[1], H_batch, W)
            j = 0
            for i in range(13):
                if (i == 12):m = 5
                else:m = 21
                x = cdf_input_tensor[j:j+m];j = j + m
                cdf_tensor = function.forward_cdf_encode(x,patches_reshaped,wms_reshaped)
                cdf_out.append(cdf_tensor.detach().cpu())
            cdfs2 = (torch.cat(cdf_out, dim=0))
            cdf_total.append(((cdfs2.reshape(257, 3, H_batch * W)).permute(2, 1, 0)).reshape(-1, 257))
            if (for_finish_flag==1):
                break
        cdfs_enc = (torch.cat(cdf_total, dim=0));cdfs_enc = torch.clamp(cdfs_enc, min=0, max=1)
        sym_org =((255 * (symbol_reshaped)).type(torch.int16)).cpu()
        sym_org=((sym_org.reshape(-1,H*W)).permute(1,0)).reshape(-1)
        index_list=get_index_list(self, H, W, K)
        cdfs_enc[:, :] = cdfs_enc[index_list, :]; sym_org[:] = sym_org[index_list]
        byte_stream, real_bits = self.codec.encode(cdfs_enc.numpy(),sym_org.numpy(),self.bistream_path)
        total_bits = len(byte_stream) * 8
        rate = total_bits /num_symbol;print("Bitrate: ", rate)
        return rate



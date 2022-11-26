from range_coder import *
import torch
from torch.nn import functional as F
import pickle

#######################################################################################
@torch.no_grad()
def calculate_batch_steps(self,H, W):
       device=self.device
       infVal = (torch.tensor([1000], dtype=torch.float32)).to(device)
       pixel_steps = torch.arange(0.5 / 255.0, 1, 1 / 255.0).to(device)
       steps = (torch.cat((-1 * infVal, pixel_steps, infVal), 0))
       return steps

############################# FOR PARALEL ENCODING ######################
def get_index(len_parallel,index_offset):
    x=3*len_parallel+index_offset
    index_list=[0] * 3*len_parallel
    index_list[0:len_parallel]=list(range(index_offset,x,3))
    index_list[len_parallel:2*len_parallel]=list(range(index_offset+1,x+1,3))
    index_list[2*len_parallel:3*len_parallel]=list(range(index_offset+2,x+2,3))
    return index_list

def get_index_list(self, H, W, K):
    time_index, length = load_paralellize_matrix(self, H, W, K)
    index = 0; l = 0; index_list = [];
    for par_index_list in time_index:
        par_index_list = list(filter(lambda x: x[0] < H, par_index_list))  # Filter list according to height
        par_index_list = list(filter(lambda x: x[1] < W, par_index_list))  # Filter list according to weight
        len_parallel=len(par_index_list)
        a = get_index(len_parallel, index)
        index_list.extend(a)
        index = index + 3 * len(par_index_list)
        if (l == length):break
        l = l + 1
    return index_list
#############################################################################
#Some parts of the following code for wavefront parallelization follows Zhang et al's codes at link https://github.com/zmtomorrow/ParallelNeLLoC

def load_paralellize_matrix(self,H,W,K):
    if (self.decoder_method == 1):
        file_name = self.matrix_file_name1
        time_length = W + int((K + 1) / 2) * (H - 1)
    elif (self.decoder_method == 2):
        file_name = self.matrix_file_name2
        time_length = H+W-1
    retreived_default_scores = pickle.load(open(file_name, 'rb'))
    return retreived_default_scores,time_length

def load_paralellize_matrix_for_encode(file_name):
    retreived_default_scores = pickle.load(open(file_name, 'rb'))
    return retreived_default_scores

def get_patches(self, img, H, W, k):
    decoder_type=self.decoder_method;mid = int(k / 2);pad = (mid, mid, mid, mid);
    img = F.pad(img, pad, "constant", 0)
    img=img.cpu().numpy()
    if (decoder_type==1): file_name = self.paralel_data_matrix1_encode
    elif (decoder_type==2):file_name = self.paralel_data_matrix2_encode
    else:print("Select valid mode!")
    time_index = load_paralellize_matrix_for_encode(file_name)
    time_index = list(filter(lambda x: x[0] < H, time_index))  # Filter list according to height
    time_index = list(filter(lambda x: x[1] < W, time_index))  # Filter list according to weight
    patch_list = []
    for i, j in time_index:
        patch=img[0, :, i:i + mid + 1, j:j + k]
        patch_list.append(patch)
    patches = np.stack(patch_list)
    if (decoder_type == 2):
        patches[:,:,0,4] = patches[:,:,0,3];patches[:,:,1,3] = patches[:,:,0,3];patches[:,:,1,4] = patches[:,:,0,3]
    patches = (torch.from_numpy(patches)).to(self.device)
    return patches
###################################################################################################################################

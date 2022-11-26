import os
import sys
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import RandomCrop, ToTensor, Compose, CenterCrop, RandomHorizontalFlip

class ImageDataLoader():
    def __init__(self, config):
        self.train_dataset = ImageDataset(config.train_data,config.patch_size,train=True)
        self.test_dataset  = ImageDataset(config.test_data,0,train=False)
        self.valid_dataset = ImageDataset(config.valid_data,config.val_patch_size,train=False)
        
        num_workers = 0 if config.mode == 'debug' else 5
        self.train_loader = DataLoader(self.train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=False)
        self.test_loader  = DataLoader(self.test_dataset, batch_size=1,shuffle=False, num_workers=0,pin_memory=True,drop_last=False)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=0, pin_memory=True,drop_last=False)

class ImageDataset(Dataset):
    def __init__(self, root, size, train=True):
        self.size = size
        try:
            if isinstance(root, str):
                self.image_files = [os.path.join(root, f)  for f in os.listdir(root) if (f.endswith('.png') or f.endswith('.jpg'))]
            else:
                self.image_files = []
                for i in range(0, len(root)):
                    self.image_files_temp = [os.path.join(root[i], f)  for f in os.listdir(root[i]) if (f.endswith('.png') or f.endswith('.jpg'))]
                    self.image_files = self.image_files + self.image_files_temp
        except:
            logging.getLogger().exception('Dataset could not be found. Drive might be unmounted.', exc_info=False)
            sys.exit(1)
        if size == 0:
            self.transforms = Compose([ToTensor()])
        else:
            crop = RandomCrop(size) if train else CenterCrop(size)
            self.transforms = Compose([crop, RandomHorizontalFlip(), ToTensor()])

    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, i):
        img = pil_loader(self.image_files[i])
        return self.transforms(img)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

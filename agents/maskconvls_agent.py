import torch
from torch import optim
from agents.base import BaseAgent
from graphs.models.MaskedConvLossless_net import MaskedConvLosslessNet
from graphs.losses.rate_distortion_loss import TrainRateLoss, ValidRateLoss
from dataloaders.image_dl import ImageDataLoader
from loggers.rate import RateLogger

class MaskedConvLosslessAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = MaskedConvLosslessNet(config)
        self.model = self.model.to(self.device)
        self.lr = self.config.learning_rate
        self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr':self.lr}])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)
        self.data_loader  = ImageDataLoader(config)
        self.train_loss   = TrainRateLoss()
        self.valid_loss   = ValidRateLoss()
        self.train_logger = RateLogger()
        self.valid_logger = RateLogger()
        self.test_logger  = RateLogger()
        if config.mode == 'test':
            self.load_checkpoint('model_best.pth.tar')
        elif config.resume_training:
            self.load_checkpoint(self.config.checkpoint_file)
        self.patch_size = config.patch_size
        self.val_patch_size = config.val_patch_size
        self.run_mode=config.run_mode


    def train_one_epoch(self):
        self.model.train()
        for batch_idx, y in enumerate(self.data_loader.train_loader):
            y = y.to(self.device)
            self.optimizer.zero_grad()# run through model, calculate loss, back-prop etc.
            self_infos = self.model(y)
            bpp_rate_loss = self.train_loss(self_infos)
            bpp_rate_loss.backward()
            self.optimizer.step()
            self.current_iteration += 1
            self.train_logger(bpp_rate_loss.item())
        self.train_logger.display(lr=self.optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, y in enumerate(self.data_loader.valid_loader):
                y = y.to(self.device)
                self_infos = self.model(y)
                bpp_rate_loss = self.valid_loss(self_infos)
                self.valid_logger(bpp_rate_loss.item())
            valid_bpp_rate_loss = self.valid_logger.display(lr=0.0)
            self.scheduler.step()
            return valid_bpp_rate_loss

    @torch.no_grad()
    def test(self):
        self.model.eval()
        with torch.no_grad():
            if self.run_mode=="encode":
                for batch_idx, y in enumerate(self.data_loader.test_loader):
                    bpp = self.model.compress(y.to(self.device))
                    print("Encoding finished..")
            elif self.run_mode == "decode":
                decoded_image = self.model.decompress(self.config.H, self.config.W)
                print("Decoding finished..")
            else:
                print("Enter a valid mode!")
import logging
import shutil
import torch
from torch.backends import cudnn
cudnn.benchmark = True
cudnn.enabled = True

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.best_valid_loss = float('inf')  # 0
        self.current_epoch = 0
        self.current_iteration = 0
        self.device = torch.device("cuda")
        self.cuda = torch.cuda.is_available()
        self.manual_seed = 1337
        torch.cuda.manual_seed(self.manual_seed)
        torch.cuda.set_device(0)
        
    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def validate_recu_reco(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=self.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_valid_loss = checkpoint['best_valid_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.logger.info("Checkpoint loaded successfully \
                            from '{}' at (epoch {}) at (iteration {})\n"
                            .format(self.config.checkpoint_dir, checkpoint['epoch'],
                            checkpoint['iteration']))

            self.model.to(self.device)
            # Fix the optimizer cuda error
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping..."
                             .format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")
            
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'best_valid_loss': self.best_valid_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(state, self.config.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'validate':
                self.validate()
            elif self.config.mode == 'validate_recu_reco':
                self.validate_recu_reco()
            elif self.config.mode == 'train':
                self.train()
            elif self.config.mode == 'debug':
                with torch.autograd.detect_anomaly():
                    self.train()
            else:
                raise NameError("'" + self.config.mode + "'" 
                                + ' is not a valid training mode.' )
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
        except AssertionError as e:
            raise e
        except Exception as e:
            self.save_checkpoint()
            raise e

    def train(self):
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            if not (self.current_epoch+1) % self.config.validate_every:
                valid_loss = self.validate()
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self.save_checkpoint(is_best=is_best)
            self.current_epoch += 1

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()

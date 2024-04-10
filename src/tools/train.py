from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config as cfg
from .evaluate import Evaluator

from src.utils.data_utils import DataUtils
from src.utils.tensorboard import Tensorboard
from src.utils.map_metrics import AverageMeter
from src.data.total_text import TotalTextDataset
from src.utils.visualization import Visualization
from src.utils.logger import logger, set_logger_tag
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss


set_logger_tag(logger, tag='TRAINING')


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_acc = 0.0
        self.create_data_loader()
        self.create_model()
        self.eval = Evaluator(self.valid_dataset, self.model)
    
    def create_data_loader(self):
        self.train_dataset = TotalTextDataset(mode="Train")
        self.valid_dataset = TotalTextDataset(mode="Eval")
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.args.batch_size, 
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)
    
    def create_model(self):
        self.model = DiffBinarization(pretrained=True, backbone=cfg['Global']['backbone']).to(self.args.device)
        self.loss_func = DiffBinarizationLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        if self.args.resume:
            logger.info("Resuming training ...")
            last_ckpt = self.args.last_ckpt_pth
            if os.path.exists(last_ckpt):
                ckpt = torch.load(last_ckpt, map_location=self.args.device)
                self.start_epoch = self.resume_training(ckpt)
                logger.info(f"Loading checkpoint with start epoch: {self.start_epoch}, best acc: {self.best_acc}")
    
    def train(self):
        metrics = {
            'shrink_maps_loss': AverageMeter(),
            'thresh_maps_loss': AverageMeter(),
            'binary_maps_loss':  AverageMeter(),
            'total_loss': AverageMeter()
        }
        self.model.train()
        for epoch in range(self.start_epoch, self.args.epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                bz = images.size(0)
                images = DataUtils.to_device(images)
                labels = DataUtils.to_device(labels)
                out = self.model(images)
                loss = self.loss_func(out, labels)
                self.optimizer.zero_grad()
                loss['total_loss'].backward()
                self.optimizer.step()

                metrics['shrink_maps_loss'].update(loss['shrink_maps_loss'])
                metrics['thresh_maps_loss'].update(loss['thresh_maps_loss'])
                metrics['binary_maps_loss'].update(loss['binary_maps_loss'])
                metrics['total_loss'].update(loss['total_loss'])
                
                print(f"Epoch {epoch} - batch {i+1}/{len(self.train_loader)} - total_loss: {metrics['total_loss'].val: .4f} - shrink_maps_loss: {metrics['shrink_maps_loss'].val: .4f} - thresh_maps_loss: {metrics['thresh_maps_loss'].val: .4f} - binary_maps_loss: {metrics['binary_maps_loss'].val: .4f}", end='\r')
                
                Tensorboard.add_scalars("train_loss", epoch, total_loss=metrics['total_loss'].avg)
            # self.lr_scheduler.step()
            logger.info(f"Epoch {epoch} - total_loss: {metrics['total_loss'].avg: .3f} - shrink_maps_loss: {metrics['shrink_maps_loss'].avg: .3f} - thresh_maps_loss: {metrics['thresh_maps_loss'].avg: .3f} - binary_maps_loss: {metrics['binary_maps_loss'].avg: .3f}")
            
            if epoch % self.args.eval_step == 0:
                accuracy = self.eval.eval_map()
                current_acc = accuracy['map'].val
                Tensorboard.add_scalars('eval_acc', epoch, acc=current_acc)
                
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    best_ckpt_path = self.args.best_ckpt_pth
                    self.save_ckpt(best_ckpt_path, self.best_acc, epoch)
            
            # Save last checkpoint
            last_ckpt_path = self.args.last_ckpt_pth
            self.save_ckpt(last_ckpt_path, self.best_acc, epoch)
            
            if cfg['Global']['debug_mode']:
                Visualization.output_debug(self.valid_dataset, cfg['Debug']['debug_idxs'], self.model, cfg['Debug']['debug_dir'], debug_type='train')
                Visualization.output_debug(self.train_dataset, cfg['Debug']['debug_idxs'], self.model, cfg['Debug']['debug_dir'], debug_type='valid')

    def save_ckpt(self, save_path, best_acc, epoch):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ckpt_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), 
            "best_acc": best_acc,
            "epoch": epoch
        }
        logger.info(f"Saving checkpoint to {save_path}")
        torch.save(ckpt_dict, save_path)

    def resume_training(self, ckpt):
        self.best_acc = ckpt['best_acc']
        start_epoch = ckpt['epoch'] + 1
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.model.load_state_dict(ckpt['model'])

        return start_epoch
    

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=cfg['Optimizer']['lr'])
    parser.add_argument("--device", type=str, default=cfg['Global']['device'])
    parser.add_argument("--debug_mode", type=str, default=cfg['Global']['debug_mode'])
    parser.add_argument("--epochs", type=int, default=cfg['Train']['loader']['epochs'])
    parser.add_argument("--resume", type=bool, default=cfg['Global']['resume_training'])
    parser.add_argument("--shuffle", type=bool, default=cfg['Train']['loader']['shuffle'])
    parser.add_argument("--eval_step", type=int, default=cfg['Train']['loader']['eval_step'])
    parser.add_argument("--batch_size", type=int, default=cfg['Train']['loader']['batch_size'])
    parser.add_argument("--num_workers", type=int, default=cfg['Train']['loader']['num_workers'])
    parser.add_argument("--last_ckpt_pth", type=str, default=cfg['Train']['checkpoint']['last_path'])
    parser.add_argument("--best_ckpt_pth", type=str, default=cfg['Train']['checkpoint']['best_path'])
    parser.add_argument("--pin_memory", type=bool, default=cfg['Train']['loader']['use_shared_memory'])
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()
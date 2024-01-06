from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config as cfg
from .evaluate import Evaluator

from src.utils.logger import Logger
from src.utils.metrics import BatchMeter
from src.utils.data_utils import DataUtils
from src.data.dataset import ICDAR2015Dataset
from src.utils.tensorboard import Tensorboard
from src.models.diff_binarization import DiffBinarization
from src.models.losses.db_loss import DiffBinarizationLoss


logger = Logger.get_logger("TRAINING")


class Trainer:
    def __init__(self, args) -> None:
        self.args = args
        self.start_epoch = 1
        self.best_acc = 0.0
        self.create_model()
        self.create_data_loader()
        self.eval = Evaluator(self.valid_dataset, self.model)
    
    def create_data_loader(self):
        self.train_dataset = ICDAR2015Dataset(mode="Train")
        self.valid_dataset = ICDAR2015Dataset(mode="Eval")
        self.train_loader = DataLoader(self.train_dataset, 
                                       batch_size=self.args.batch_size, 
                                       shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers,
                                       pin_memory=self.args.pin_memory)
    
    def create_model(self):
        self.model = DiffBinarization().to(self.args.device)
        self.loss_func = DiffBinarizationLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)

        if self.args.resume:
            logger.info("Resuming training ...")
            last_ckpt = self.args.last_ckpt_pth
            if os.path.exists(last_ckpt):
                ckpt = torch.load(last_ckpt, map_location=self.args.device)
                self.start_epoch = self.resume_training(ckpt)
                logger.info(f"Loading checkpoint with start epoch: {self.start_epoch}, best acc: {self.best_acc}")
    
    
    def train(self):
        metrics = {
            'shrink_maps_loss': BatchMeter(),
            'thresh_maps_loss': BatchMeter(),
            'binary_maps_loss':  BatchMeter(),
            'total_loss': BatchMeter()
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
                
                print(f"Epoch {epoch} - batch {i+1}/{len(self.train_loader)} - total_loss: {metrics['total_loss'].get_value(): .4f} - shrink_maps_loss: {metrics['shrink_maps_loss'].get_value(): .4f} - thresh_maps_loss: {metrics['thresh_maps_loss'].get_value(): .4f} - binary_maps_loss: {metrics['binary_maps_loss'].get_value(): .4f}", end='\r')
                
                Tensorboard.add_scalars("train_loss", epoch, total_loss=metrics['total_loss'].get_value("mean"))

            logger.info(f"Epoch {epoch} - total_loss: {metrics['total_loss'].get_value('mean'): .3f} - shrink_maps_loss: {metrics['shrink_maps_loss'].get_value('mean'): .3f} - thresh_maps_loss: {metrics['thresh_maps_loss'].get_value('mean'): .3f} - binary_maps_loss: {metrics['binary_maps_loss'].get_value('mean'): .3f}")
            
            if epoch % self.args.eval_step == 0:
                accuracy = self.eval.eval()
                current_acc = accuracy['map_50'].get_value('mean')
                Tensorboard.add_scalars('eval_loss', epoch, loss=current_acc)
                Tensorboard.add_scalars('eval_acc', epoch, loss=accuracy['map_50'].get_value('mean'))

                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    best_ckpt_path = self.args.best_ckpt_pth
                    self.save_ckpt(best_ckpt_path, self.best_acc, epoch)
            
            # Save last checkpoint
            last_ckpt_path = self.args.last_ckpt_pth
            self.save_ckpt(last_ckpt_path, self.best_acc, epoch)

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
    parser.add_argument("--lr", default=cfg['Optimizer']['lr'], type=float)
    parser.add_argument("--device", default=cfg['Global']['device'], type=str)
    parser.add_argument("--resume", default=cfg['Global']['resume_training'])
    parser.add_argument("--epochs", default=cfg['Train']['loader']['epochs'], type=int)
    parser.add_argument("--shuffle", default=cfg['Train']['loader']['shuffle'], type=bool)
    parser.add_argument("--eval_step", default=cfg['Train']['loader']['eval_step'], type=int)
    parser.add_argument("--batch_size", default=cfg['Train']['loader']['batch_size'], type=int)
    parser.add_argument("--num_workers", default=cfg['Train']['loader']['num_workers'], type=int)
    parser.add_argument("--pin_memory", default=cfg['Train']['loader']['use_shared_memory'], type=bool)
    parser.add_argument("--last_ckpt_pth", default=cfg['Train']['checkpoint']['last_path'], type=str)
    parser.add_argument("--best_ckpt_pth", default=cfg['Train']['checkpoint']['best_path'], type=str)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    trainer = Trainer(args)
    trainer.train()
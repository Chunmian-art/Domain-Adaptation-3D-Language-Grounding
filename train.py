import os
from pathlib import Path
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import random
import torch
import models
from data.dataset_resnet import CLIPGraspingDataset
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime


@hydra.main(config_path="cfgs", config_name="train")
def main(cfg):
    # set random seeds
    model_name = cfg['train']['model']
    currentDateAndTime = datetime.now()
    logger = TensorBoardLogger(f"/")
    seed = cfg['train']['random_seed']
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    hydra_dir = Path(os.getcwd())
    checkpoint_path = hydra_dir
    last_checkpoint = cfg['train']['pretrained_model']

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=checkpoint_path,
        filename='{epoch:04d}-{val_acc:.5f}',
        save_top_k=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=15, verbose=False, mode="max")
    trainer = Trainer(
        gpus=[0],
        # accelerator='ddp_cuda',
        fast_dev_run=cfg['debug'],
        checkpoint_callback=checkpoint_callback,
        max_epochs=cfg['train']['max_epochs'],
        logger=logger
    )

    # dataset
    train = CLIPGraspingDataset(cfg, mode='train')
    valid = CLIPGraspingDataset(cfg, mode='valid')
    test = CLIPGraspingDataset(cfg, mode='test')

    # model
    model = models.names[cfg['train']['model']](cfg, train, valid)

    # resume epoch and global_steps
    if os.path.exists(cfg['train']['pretrained_model']):
        last_ckpt = torch.load(last_checkpoint)
        print(f"================================== {last_checkpoint} is loaded ==================================")
        # trainer.current_epoch = last_ckpt['epoch']
        trainer.current_epoch = 0
        trainer.global_step = last_ckpt['global_step']
        model.load_state_dict(last_ckpt["state_dict"], strict=False)
        del last_ckpt

    # with torch.autograd.detect_anomaly():
    trainer.fit(
        model,
        train_dataloader=DataLoader(train, batch_size=cfg['train']['batch_size'], num_workers=8),
        val_dataloaders=DataLoader(valid, batch_size=cfg['train']['batch_size'], num_workers=8),
    )

    trainer.test(
        model=model,
        test_dataloaders=DataLoader(test, batch_size=cfg['train']['batch_size'], num_workers=8),
        ckpt_path='best'
    )


if __name__ == "__main__":
    main()
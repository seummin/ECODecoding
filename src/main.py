# -*- coding: utf-8 -*-

import os
import hydra

import wandb
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import GPT2Tokenizer

from datamodule import *
from learner import *


@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg):
    seed_everything(cfg.seed)

    tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialoGPT-small")

    datamodule = DataModule(cfg, tokenizer)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    ckpt_path = cfg.ckpt_path = (
        f"../../experiment/{cfg.method}/{cfg.datamodule.data_name}/"
        f"{cfg.datamodule.batch_size * cfg.gradient_accumulation_steps}/"
        f"{cfg.learning_rate}/{cfg.seed}/{cfg.epochs}/{cfg.learner.pre_seq_len}"
    )
    cfg.learner.ckpt_path = ckpt_path
    run_name = (
        f"{cfg.method}_{cfg.seed}_"
        f"{cfg.datamodule.batch_size * cfg.gradient_accumulation_steps}_"
        f"{cfg.learning_rate}_{cfg.epochs}_{cfg.learner.pre_seq_len}"
    )
    cfg.learner.data_name = cfg.datamodule.data_name

    if cfg.learner.data_name is None:
        print("Learner data_name is not defined")
        raise ValueError

    print(cfg.method, run_name)
    print(cfg.learner.ckpt_path)

    if not os.path.exists(cfg.wandb_dir):
        os.makedirs(cfg.wandb_dir)

    logger = WandbLogger(
        name=f"{cfg.datamodule.data_name}_{run_name}",
        save_dir=cfg.wandb_dir,
        project="CTRL_DialoGPT_ver1",
    )

    os.makedirs(cfg.ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.ckpt_path,
        filename="best_model",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=cfg.devices,
        logger=logger,
        max_steps=-1,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        callbacks=[checkpoint_callback],
    )

    learner = Learner(cfg.learner, tokenizer)

    if cfg.test is True:
        trainer.test(learner, test_loader, ckpt_path=os.path.join(ckpt_path, "best_model.ckpt"))
    else:
        trainer.fit(learner, train_loader, val_loader)
        trainer.test(learner, test_loader, ckpt_path="best")

    wandb.finish()


if __name__ == "__main__":
    main()

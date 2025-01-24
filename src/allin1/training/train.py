import hydra
import lightning
from lightning import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from .data import HarmonixDataModule
from .trainer import AllInOneTrainer
from .evaluate import evaluate
from ..models import load_pretrained_model  # Import the function to load pretrained models
from ..config import Config


@hydra.main(version_base=None, config_name='config')
def main(cfg: Config):
    makeup_config(cfg)

    # Setting all random seeds
    lightning.seed_everything(cfg.seed)

    if cfg.data.name == 'files':
        dm = HarmonixDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}")

    print("=> Initializing model...")
    # Initialize model with pretrained weights from harmonix-fold0
    pretrained_model = load_pretrained_model(
        model_name="harmonix-fold0",  # Specify the harmonix-fold0 pretrained model
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Pass the pretrained model to the trainer
    model = AllInOneTrainer(cfg)
    model.model.load_state_dict(pretrained_model.state_dict(), strict=False)  # Transfer weights to the trainer's model

    print("=> Pretrained weights loaded.")

    # Setup Wandb logger
    wandb_logger = WandbLogger(
        project='models',
        tags=[
            f'fold{cfg.fold}'
        ] + ([cfg.case] if cfg.case else []),
        log_model=False if cfg.debug or cfg.sanity_check or cfg.offline else 'all',
        offline=cfg.debug or cfg.sanity_check or cfg.offline,
    )
    wandb_logger.log_hyperparams(cfg)
    wandb_logger.experiment.define_metric("val/loss", summary="min")

    # Add callbacks
    callbacks = [
        ModelCheckpoint(monitor="val/loss", mode="min"),
        EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=cfg.early_stopping_patience,
            min_delta=1e-4,
            log_rank_zero_only=True,
            verbose=True,
        ),
        LearningRateMonitor(),
    ]

    if cfg.swa_lr > 1e-4:
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.swa_lr))

    # Initialize Trainer
    trainer = Trainer(
        accelerator="cpu" if cfg.debug else "auto",
        devices=1,
        gradient_clip_val=cfg.gradient_clip,
        logger=wandb_logger,
        callbacks=None if cfg.sanity_check else callbacks,
        check_val_every_n_epoch=cfg.validation_interval_epochs,
        max_epochs=cfg.max_epochs,
        fast_dev_run=cfg.debug and not cfg.sanity_check,
        overfit_batches=cfg.sanity_check_size if cfg.sanity_check else 0,
    )

    if trainer.is_global_zero:
        print("=" * 80)
        print("Config")
        print("=" * 80)
        print(OmegaConf.to_yaml(cfg))
        print("=" * 80)

    # Start training
    trainer.fit(
        model=model,
        datamodule=dm,
    )
    print("=> Finished training.")

    # Run evaluation
    if trainer.is_global_zero:
        print("=> Running evaluation...")
        evaluate(
            model=model,
            trainer=trainer,
        )


def makeup_config(cfg: Config):
    if cfg.sanity_check:
        cfg.sched = None
        cfg.warmup_epochs = 0
        cfg.weight_decay = 0
        cfg.drop_conv = 0
        cfg.drop_path = 0
        cfg.drop_hidden = 0
        cfg.drop_attention = 0
        cfg.validation_interval_epochs = 50


if __name__ == '__main__':
    main()

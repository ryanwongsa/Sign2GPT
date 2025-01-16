from ignite.handlers import CosineAnnealingScheduler, LRScheduler
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import math
from ignite.engine import Engine, Events
from ignite.handlers.param_scheduler import ReduceLROnPlateauScheduler
from scheduler.reducelr import CustomReduceLROnPlateauScheduler


def prep_scheduler(scheduler_name, cfg, train_dl, optimizer, trainer, evaluator):
    if scheduler_name == "cosine":
        epoch_length = len(train_dl)
        num_cycles = 1
        cycle_size = math.ceil((cfg.max_epochs * epoch_length) / num_cycles)
        scheduler = CosineAnnealingScheduler(
            optimizer,
            "lr",
            start_value=cfg.lr_scheduler_params["start_lr"],
            end_value=cfg.lr_scheduler_params["end_lr"],
            cycle_size=cycle_size,
        )
        return scheduler, "trainer", Events.ITERATION_STARTED
    elif scheduler_name == "warmupwithcosine":
        from scheduler.warmup_with_cosine import get_warmup_with_cosine

        scheduler = get_warmup_with_cosine(cfg, optimizer, train_dl)
        if isinstance(scheduler, list) and len(scheduler) == 1:
            scheduler = scheduler[0]
        return scheduler, "trainer", Events.ITERATION_STARTED
    elif scheduler_name == "step":
        lr_scheduler = MultiStepLR(optimizer=optimizer, **cfg.lr_scheduler_params)
        scheduler = LRScheduler(lr_scheduler)
        return scheduler, "trainer", Events.EPOCH_STARTED
    elif scheduler_name == "reduceonplateau":
        scheduler = ReduceLROnPlateauScheduler(
            optimizer,
            cfg.lr_scheduler_params["name"],
            save_history=False,
            mode=cfg.lr_scheduler_params["mode"],
            factor=cfg.lr_scheduler_params["factor"],
            patience=cfg.lr_scheduler_params["patience"],
            threshold_mode=cfg.lr_scheduler_params["threshold_mode"],
            threshold=cfg.lr_scheduler_params["threshold"],
            trainer=None,
            cooldown=cfg.lr_scheduler_params["cooldown"],
            verbose=cfg.lr_scheduler_params["verbose"],
        )

        return scheduler, "evaluator", Events.COMPLETED
    elif scheduler_name == "customreduceonplateau":
        scheduler = CustomReduceLROnPlateauScheduler(
            optimizer,
            cfg.lr_scheduler_params["name"],
            mode=cfg.lr_scheduler_params["mode"],
            factor=cfg.lr_scheduler_params["factor"],
            patience=cfg.lr_scheduler_params["patience"],
            threshold_mode=cfg.lr_scheduler_params["threshold_mode"],
            threshold=cfg.lr_scheduler_params["threshold"],
            cooldown=cfg.lr_scheduler_params["cooldown"],
            verbose=cfg.lr_scheduler_params["verbose"],
        )

        return scheduler, "evaluator", Events.COMPLETED
    elif scheduler_name == "none":
        return None, None, None

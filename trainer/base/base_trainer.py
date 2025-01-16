from ignite.metrics import Average
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
import torch

from train_utils.seed_helpers import setup_seed
from augmentation.get_aug import get_aug
from dataloaders.get_dataset import get_dataset
from losses.get_loss import get_loss
from optimizer.get_optimizer import get_optim


class BaseTrainer:
    def __init__(self, args):
        cfg = args[0]
        self.cfg = cfg
        setup_seed(cfg.seed)

        self.support_bfloat = torch.cuda.is_bf16_supported()
        self.dtype = torch.bfloat16 if self.support_bfloat else torch.float16

    def get_dataloaders(self, cfg):
        transform = get_aug(cfg.aug_name, cfg.aug_params)
        self.transform = transform
        train_dl, train_dict = get_dataset(
            cfg.train_ds_name, cfg.train_ds_params, transform=transform
        )
        valid_dl, valid_dict = get_dataset(
            cfg.valid_ds_name, cfg.valid_ds_params, transform=transform
        )
        test_dl, test_dict = get_dataset(
            cfg.test_ds_name, cfg.test_ds_params, transform=transform
        )
        return train_dl, valid_dl, test_dl, train_dict, valid_dict, test_dict

    def loss_fn(self, y_pred, target):
        loss, dict_losses = self.criterion(y_pred, target)
        return loss, dict_losses

    def init_criterion(self, cfg):
        criterion = get_loss(cfg.criterion_name, cfg.criterion_params)
        self.criterion = criterion.to(idist.device())

    def init_optimizer(self, cfg):
        self.optimizer = get_optim(cfg.optimizer_name, cfg.optimizer_params, self.model)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("MODEL:", count_parameters(self.model))

        if not self.support_bfloat:
            self.scaler = torch.cuda.amp.GradScaler()  # growth_interval=500)
        else:
            self.scaler = None
        self.grad_clip_norm = cfg.grad_clip_norm if ("grad_clip_norm" in cfg) else None
        self.grad_clip_value = (
            cfg.grad_clip_value if ("grad_clip_value" in cfg) else None
        )

    def get_only_engines(self):
        trainer = Engine(self.train_step)
        evaluator = Engine(self.eval_step)
        valid_tester = Engine(self.test_step)
        test_tester = Engine(self.test_step)
        return trainer, evaluator, valid_tester, test_tester

    def get_only_dataloaders(self):
        (
            train_dl,
            valid_dl,
            test_dl,
            train_dict,
            valid_dict,
            test_dict,
        ) = self.get_dataloaders(self.cfg)
        return train_dl, valid_dl, test_dl

    def dict_metric_from_list(
        self, engine_type, list_of_metrics, dict_metrics, **kwargs
    ):
        return dict_metrics

    def init_metrics(self, engine, engine_type, dict_metrics={}, additional=True):
        if additional:

            def loss_display(k):
                def loss_fn(x):
                    return x["losses"][k]

                return loss_fn

            for k in ["loss"] + [f"loss_{comp}" for comp in self.criterion.crit_keys]:
                dict_metrics[f"{engine_type}/avg_{k}"] = Average(
                    output_transform=loss_display(k)
                )

        for k, v in dict_metrics.items():
            v.attach(engine, k)

    def prep_scheduler(self, cfg, train_dl, optimizer, trainer, evaluator):
        from scheduler.get_scheduler import prep_scheduler

        scheduler, engine_name, event_name = prep_scheduler(
            cfg.lr_scheduler, cfg, train_dl, optimizer, trainer, evaluator
        )

        if scheduler:
            if isinstance(scheduler, list):
                for sch in scheduler:
                    if engine_name == "trainer":
                        trainer.add_event_handler(event_name, sch)
                    elif engine_name == "evaluator":
                        evaluator.add_event_handler(event_name, sch)
            else:
                if engine_name == "trainer":
                    trainer.add_event_handler(event_name, scheduler)
                elif engine_name == "evaluator":
                    evaluator.add_event_handler(event_name, scheduler)
        return scheduler

    def save_checkpoints(
        self, cfg, trainer, evaluator, score_function, best_only=False, to_save={}
    ):
        if to_save == {}:
            to_save = {
                "trainer": trainer,
                "optimizer": self.optimizer,
                "model": self.model,
            }
        if self.scaler is not None:
            to_save["scaler"] = self.scaler

        if self.scheduler is not None:
            to_save["scheduler"] = self.scheduler

        checkpoint = Checkpoint(
            to_save,
            DiskSaver(dirname=cfg.save_dir, require_empty=False),
            n_saved=1,
            filename_prefix=f"best_result",
            score_function=score_function,
            score_name=None,
            global_step_transform=global_step_from_engine(trainer),
            greater_or_equal=True,
        )

        latest_checkpoint = Checkpoint(
            to_save,
            DiskSaver(dirname=cfg.save_dir, require_empty=False),
            n_saved=1,
            filename_prefix=f"latest_epoch",
        )

        def before_latest_checkpoint(engine):
            engine.state.output = None
            engine.state.batch = None

        if cfg.save_ckpt:
            evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint)
            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=1), before_latest_checkpoint
            )
            if not best_only:
                trainer.add_event_handler(
                    Events.EPOCH_COMPLETED(every=1), latest_checkpoint
                )

    def load_checkpoints(self, cfg, trainer, objects_to_load={}):
        if objects_to_load == {}:
            objects_to_load = {
                "model": self.model,
                "optimizer": self.optimizer,
                "trainer": trainer,
            }
            if self.scaler is not None:
                objects_to_load["scaler"] = self.scaler

        if cfg.model_checkpoint_dir != "":
            print("LOADING MODEL FROM:", cfg.model_checkpoint_dir)
            objects_to_load = {"model": self.model}
            Checkpoint.load_objects(
                to_load=objects_to_load,
                # checkpoint=cfg.model_checkpoint_dir,
                checkpoint=torch.load(
                    cfg.model_checkpoint_dir, map_location="cpu"
                ),
                strict=False,
            )

        if cfg.resume:
            from train_utils.checkpoint_helpers import (
                get_latest_saved_file,
                get_best_checkpoint_details,
            )

            latest_checkpoint_from_resume = get_latest_saved_file(
                cfg.save_dir, extension="pt", name_latest="latest_epoch"
            )
            if latest_checkpoint_from_resume[0]:
                if self.scheduler:
                    objects_to_load["scheduler"] = self.scheduler

                Checkpoint.load_objects(
                    to_load=objects_to_load,
                    checkpoint=torch.load(
                        latest_checkpoint_from_resume[0], map_location="cpu"
                    ),
                )

    def cleaning_with_progress(self, trainer, evaluator, cfg, train_dl):
        if cfg.train_length is not None:
            tstep = cfg.train_length // 10
        else:
            tstep = len(train_dl) // 10
        from datetime import datetime

        def progress_train(engine):
            if idist.get_rank() == 0:
                compl = "{0:.2f}".format(
                    (engine.state.iteration % engine.state.epoch_length)
                    / engine.state.epoch_length
                )
                n = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{n}] Epoch {engine.state.epoch}: ({compl}) - {(engine.state.iteration % engine.state.epoch_length)}/{engine.state.epoch_length}"
                )

        # self.logger.apply_gradient_log(trainer, self.model, tstep, self.scaler)

        if cfg.train_length is not None:
            tstep = max(cfg.train_length // 5, 50)
        else:
            tstep = max(len(train_dl) // 5, 50)

        if "pbar" in cfg.logger_name:
            if idist.get_rank() == 0:
                pbar = ProgressBar()
                pbar.attach(evaluator)

            if idist.get_rank() == 0:
                pbar = ProgressBar()
                pbar.attach(trainer)
        else:
            trainer.add_event_handler(
                Events.ITERATION_COMPLETED(every=tstep), progress_train
            )
        # -------------------------------------- [END] LOGGING HANDLER --------------------------------------
        from ignite.contrib.engines.common import empty_cuda_cache

        def clear_states_engines(engine):
            trainer.state.output = None
            trainer.state.batch = None
            engine.state.output = None
            engine.state.batch = None

        trainer.add_event_handler(Events.EPOCH_STARTED, clear_states_engines)
        evaluator.add_event_handler(Events.EPOCH_STARTED, clear_states_engines)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)

        if idist.get_world_size() > 1:

            def set_epoch(engine):
                train_dl.sampler.set_epoch(engine.state.epoch)

            trainer.add_event_handler(Events.EPOCH_STARTED, set_epoch)

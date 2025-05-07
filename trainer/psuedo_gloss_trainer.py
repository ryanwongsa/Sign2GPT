import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor

from models.get_models import get_model

from trainer.base.base_trainer import BaseTrainer
from callbacks.full_callback import LoggingCallback
from train_utils.checkpoint_helpers import (
    get_latest_saved_file,
    get_best_checkpoint_details,
)


class Trainer(BaseTrainer):
    def __init__(self, local_rank, *args, **kwargs):
        super().__init__(args)
        cfg = args[0]
        if "run" in cfg:
            pass
        else:
            self.logger = LoggingCallback(self.cfg)
            self.logger.start_logger()

        self.max_length = 128

        (
            train_dl,
            valid_dl,
            test_dl,
            train_dict,
            valid_dict,
            test_dict,
        ) = self.get_dataloaders(cfg)

        self.init_models()
        self.init_criterion(cfg)
        self.init_optimizer(cfg)

        trainer = Engine(self.train_step)
        evaluator = Engine(self.eval_step)
        valid_tester = Engine(self.test_step)
        test_tester = Engine(self.test_step)

        self.scheduler = self.prep_scheduler(
            cfg, train_dl, self.optimizer, trainer, evaluator
        )

        self.init_metrics(
            trainer,
            "train",
            [
                "accuracy",
                "class_accuracy",
                "f1_score",
                "class_f1_score",
            ],  
            additional=True,
            max_length=self.max_length,
            num_samples=train_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )
        self.init_metrics(
            evaluator,
            "valid",
            [
                "accuracy",
                "class_accuracy",
                "f1_score",
                "class_f1_score",
            ], 
            additional=True,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )

        self.init_metrics(
            valid_tester,
            "valid_test",
            [
                "accuracy",
                "class_accuracy",
                "f1_score",
                "class_f1_score",
            ],
            additional=False,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )
        self.init_metrics(
            test_tester,
            "test_test",
            [
                "accuracy",
                "class_accuracy",
                "f1_score",
                "class_f1_score",
            ], 
            additional=False,
            max_length=self.max_length,
            num_samples=test_dict["length"],
            num_classes=len(train_dict["dict_lem_to_id"]),
        )

        def score_function(engine):
            return cfg.score_factor * float(engine.state.metrics[cfg.score_name])

        to_save = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer": trainer,
        }
        if self.scaler is not None:
            to_save["scaler"] = self.scaler
        if self.scheduler is not None:
            to_save["scheduler"] = self.scheduler
        self.save_checkpoints(
            cfg, trainer, evaluator, score_function, best_only=False, to_save=to_save
        )
        objects_to_load = {
            "model": self.model,
            "optimizer": self.optimizer,
            "trainer": trainer,
        }
        if self.scaler is not None:
            objects_to_load["scaler"] = self.scaler
        if self.scheduler is not None:
            objects_to_load["scheduler"] = self.scheduler

        if "model_only" in cfg and cfg.model_only == True:
            objects_to_load = {"model": self.model}
        self.load_checkpoints(cfg, trainer, objects_to_load=objects_to_load)

        if "run" in cfg:
            if cfg["load_from_ckpt"] == "best":
                ckpt, _, _ = get_best_checkpoint_details(
                    cfg.save_dir, best_checkpoint_name="_result_checkpoint_"
                )
                self.model.load_state_dict(
                    torch.load(ckpt, map_location="cpu")["model"]
                )
            elif cfg["load_from_ckpt"] == "latest":
                ckpt, _, _ = get_latest_saved_file(
                    cfg.save_dir, extension="pt", name_latest="latest_epoch"
                )
                self.model.load_state_dict(
                    torch.load(ckpt, map_location="cpu")["model"]
                )
            else:
                print("SKIPPING THE LOADING FROM CHECKPOINT")
            self.trainer = trainer
            self.evaluator = evaluator
            self.valid_tester = valid_tester
            self.test_tester = test_tester
        else:
            self.prepare_runner(
                cfg, trainer, evaluator, valid_dl, valid_tester, test_tester, test_dl
            )

            self.cleaning_with_progress(trainer, evaluator, cfg, train_dl)

            trainer.run(
                train_dl, max_epochs=cfg.max_epochs, epoch_length=cfg.train_length
            )

    def prepare_runner(
        self, cfg, trainer, evaluator, valid_dl, valid_tester, test_tester, test_dl
    ):
        def run_evaluator(engine):
            engine.state.output = None
            engine.state.batch = None
            evaluator.run(valid_dl, max_epochs=1, epoch_length=cfg.val_length)

        self.logger.on_train_epoch_end(trainer, self.optimizer)
        self.logger.on_train_iteration(trainer, self.model, self.scaler)

        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), run_evaluator)
        self.logger.on_valid_epoch_end(trainer, evaluator)

        self.logger.on_completion(trainer)

    def dict_metric_from_list(
        self, engine_type, list_of_metrics, dict_metrics, **kwargs
    ):
        def output_fn(a):
            x = a["y_pred"]["dict_post_output"]["logits"]
            tgt = a["target"]["targets"]["pseudo_gloss_ids"]

            return (x, tgt)

        if "accuracy" in list_of_metrics:
            from metrics.accuracy_score import AccuracyScore

            dict_metrics[f"{engine_type}/avg_acc"] = AccuracyScore(
                output_transform=output_fn, thresholds=0.5
            )
        if "class_accuracy" in list_of_metrics:
            from metrics.class_accuracy_score import ClassAccuracyScore

            dict_metrics[f"{engine_type}/cls_acc"] = ClassAccuracyScore(
                output_transform=output_fn,
                thresholds=0.5,
                num_classes=kwargs["num_classes"],
            )
        if "f1_score" in list_of_metrics:
            from metrics.f1_score import F1Score

            dict_metrics[f"{engine_type}/f1_score"] = F1Score(
                output_transform=output_fn, thresholds=0.5
            )

        if "class_f1_score" in list_of_metrics:
            from metrics.class_f1_score import ClassF1Score

            dict_metrics[f"{engine_type}/class_f1_score"] = ClassF1Score(
                output_transform=output_fn,
                thresholds=0.5,
                num_classes=kwargs["num_classes"],
            )

        return dict_metrics

    def init_metrics(
        self, engine, engine_type, list_of_metrics, additional=True, **kwargs
    ):
        dict_metrics = {}

        dict_metrics = self.dict_metric_from_list(
            engine_type, list_of_metrics, dict_metrics, **kwargs
        )
        super().init_metrics(
            engine, engine_type, dict_metrics=dict_metrics, additional=additional
        )

    def init_models(
        self,
    ):
        dict_model_params = self.cfg.model_params.to_dict()

        self.model = get_model(self.cfg.model_name, dict_model_params)
        self.model = idist.auto_model(
            self.model, find_unused_parameters=False, sync_bn=True
        )

    def prep_batch(self, batch, isValid=False, cuda=True):
        idx, frames = (
            batch["index"],
            batch["frames"],
        )

        frame_features = frames

        res = {
            "model_input": {
                "frame_features": frame_features,
                "max_len": torch.tensor(
                    self.cfg.max_seq_len if "max_seq_len" in self.cfg else 512
                ),
            },
            "targets": {
                "pseudo_gloss_ids": batch["pseudo_gloss_ids"]
                if "pseudo_gloss_ids" in batch
                else [],
                "index": torch.stack(idx),
            },
        }

        return (
            convert_tensor(res, device=idist.device(), non_blocking=True)
            if cuda
            else res
        )

    def train_step(self, engine, batch):
        engine.state.batch = None
        engine.state.output = None
        self.model.train()
        x = self.prep_batch(batch, isValid=False)

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            self.optimizer.zero_grad()

            y_pred = self.model(**x["model_input"])

        loss, dict_losses = self.loss_fn(y_pred, x["targets"])

        if self.scaler is not None:
            self.scaler.scale(loss).backward()

            if self.grad_clip_value is not None or self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)

                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_value_(
                        self.model.parameters(), self.grad_clip_value
                    )
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm
                    )
            self.manually_update_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.grad_clip_value
                )
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
            # ------------------------------------------------------------------------
            self.manually_update_gradients()
            # ------------------------------------------------------------------------
            self.optimizer.step()
        return {
            "y_pred": y_pred,
            "target": x,
            "losses": {
                "loss": loss.detach(),
                **{k: v.detach() for k, v in dict_losses.items()},
            },
        }

    def manually_update_gradients(self):
        pass

    def eval_step(self, engine, batch):
        engine.state.batch = None
        engine.state.output = None
        self.model.eval()
        with torch.inference_mode(True):
            x = self.prep_batch(batch, isValid=True)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                y_pred = self.model(**x["model_input"])

            loss, dict_losses = self.loss_fn(y_pred, x["targets"])

            return {
                "y_pred": y_pred,
                "target": x,
                "losses": {
                    "loss": loss.detach(),
                    **{k: v.detach() for k, v in dict_losses.items()},
                },
            }

    def test_step(self, engine, batch):
        engine.state.batch = None
        engine.state.output = None
        self.model.eval()
        with torch.inference_mode(True):
            x = self.prep_batch(batch, isValid=True)
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                y_pred = self.model(**x["model_input"])

            loss, dict_losses = self.loss_fn(y_pred, x["targets"])

            return {
                "y_pred": y_pred,
                "target": x,
                "losses": {
                    "loss": loss.detach(),
                    **{k: v.detach() for k, v in dict_losses.items()},
                },
            }

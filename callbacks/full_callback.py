import os
import ignite.distributed as idist
import json
from ignite.engine import Engine, Events
import pprint
import collections
import time
import numpy as np
import torch
import io
import pickle
from pathlib import Path


class LoggingCallback:
    def __init__(self, cfg):
        self.cfg = cfg

    def start_logger(
        self,
    ):
        from environment_variables import CONFIG

        for key, value in CONFIG.items():
            os.environ[key] = value

        self.writer = None
        self.discord_hook = None
        if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
            from train_utils.discord_hook import DiscordHook

            self.discord_hook = DiscordHook(
                title=self.cfg.name,
                project_name=self.cfg.project_name,
                discord_url=os.environ["DISCORD_URL"],
            )
        if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
            import wandb

            self.writer = wandb.init(
                project=self.cfg.project_name,
                id=self.cfg.name,
                config=json.loads(self.cfg.to_json_best_effort()),
                save_code=False,
                tags=[],
                name=self.cfg.name,
                resume="allow",
                allow_val_change=True,
            )

    def on_train_epoch_end(self, trainer, optimizer):
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def print_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {f"{k}_{k_i}": v_i, f"epoch": engine.state.epoch}
                            )
                    else:
                        self.writer.log({f"{k}": v, f"epoch": engine.state.epoch})
            if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
                results = []
                for k, v in collections.OrderedDict(
                    sorted(engine.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results.append({"name": k, "value": v, "inline": True})
                self.discord_hook.send_message(
                    content=None,
                    description=f"Training Results for Epoch: {engine.state.epoch}",
                    results=results,
                    img=None,
                )


            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                print("TRAINER", engine.state.epoch)
                dict_res = {}
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            dict_res[f"{k}_{k_i}"] = v_i
                    else:
                        dict_res[f"{k}"] = v

                pprint.pprint(dict_res)

        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def print_lr(engine):
            for i, pg in enumerate(optimizer.param_groups):
                if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                    self.writer.log(
                        {
                            f"train/lr_{i}": optimizer.param_groups[i]["lr"],
                            f"epoch": engine.state.epoch,
                        }
                    )
                if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                    pprint.pprint({f"train/lr_{i}": optimizer.param_groups[i]["lr"]})

    def on_train_iteration(self, trainer, model, scaler):
        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def print_iter_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.output["losses"].items():
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {
                                    f"{k}_{k_i}": v_i,
                                    f"global_step": engine.state.iteration,
                                }
                            )
                    else:
                        self.writer.log(
                            {f"{k}": v, f"global_step": engine.state.iteration}
                        )
            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                pprint.pprint(
                    {
                        **engine.state.output["losses"],
                        f"global_step": engine.state.iteration,
                    }
                )

        @trainer.on(Events.ITERATION_COMPLETED(every=self.cfg.log_every))
        def log_gradients(engine):
            if idist.get_rank() == 0:
                batch_norm = []
                for param in model.parameters():
                    if param.requires_grad == True:
                        if param.grad is not None:
                            batch_norm.append(param.grad.float().cpu().numpy().max())
                gradients = np.mean(batch_norm)
            if (
                "wandb" in self.cfg.logger_name
                and idist.get_rank() == 0
                and scaler is not None
            ):
                self.writer.log(
                    {
                        f"gradients": gradients,
                        f"scaler": scaler.get_scale(),
                        f"global_step": engine.state.iteration,
                    }
                )
            elif "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                self.writer.log(
                    {
                        f"gradients": gradients,
                        f"global_step": engine.state.iteration,
                    }
                )
            if (
                "text" in self.cfg.logger_name
                and idist.get_rank() == 0
                and scaler is not None
            ):
                pprint.pprint(
                    {
                        f"gradients": gradients,
                        f"scaler": scaler.get_scale(),
                        f"global_step": engine.state.iteration,
                    }
                )
            elif "text" in self.cfg.logger_name and idist.get_rank() == 0:
                pprint.pprint(
                    {
                        f"gradients": gradients,
                        f"global_step": engine.state.iteration,
                    }
                )

        if (
            "watch_grad" in self.cfg
            and self.cfg.watch_grad == True
            and self.writer is not None
        ):
            self.writer.watch(model, log_freq=500)

    def on_valid_epoch_end(self, trainer, evaluator):
        @evaluator.on(Events.EPOCH_COMPLETED(every=1))
        def print_eval_output(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            self.writer.log(
                                {f"{k}_{k_i}": v_i, f"epoch": trainer.state.epoch}
                            )
                    else:
                        self.writer.log({f"{k}": v, f"epoch": trainer.state.epoch})
            if "discord" in self.cfg.logger_name and idist.get_rank() == 0:
                results_train = []
                for k, v in collections.OrderedDict(
                    sorted(trainer.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results_train.append({"name": k, "value": v, "inline": True})

                results_valid = []
                for k, v in collections.OrderedDict(
                    sorted(engine.state.metrics.items())
                ).items():
                    if "summary" in k:
                        continue
                    if type(v) != dict:
                        results_valid.append({"name": k, "value": v, "inline": True})

                results = []
                for rt, rv in zip(results_train, results_valid):
                    name = rt["name"].replace("train_", "").replace("valid_", "")
                    rtv = "{:.4f}".format(round(rt["value"], 4))
                    rvv = "{:.4f}".format(round(rv["value"], 4))
                    results.append(
                        {
                            "name": name,
                            "value": f"{rtv} / {rvv}",
                            "inline": True,
                        }
                    )
                self.discord_hook.send_message(
                    content=None,
                    description=f"Results for Epoch: {trainer.state.epoch}",
                    results=results,
                    img=None,
                    color=16711680,
                )
            if "text" in self.cfg.logger_name and idist.get_rank() == 0:
                dict_res = {}
                for k, v in engine.state.metrics.items():
                    if "summary" in k:
                        continue
                    if type(v) == dict:
                        for k_i, v_i in v.items():
                            dict_res[f"{k}_{k_i}"] = v_i
                    else:
                        dict_res[f"{k}"] = v

                pprint.pprint({"epoch": trainer.state.epoch, **dict_res})


    def on_completion(self, trainer):
        @trainer.on(Events.COMPLETED)
        def finish_logging(engine):
            if "wandb" in self.cfg.logger_name and idist.get_rank() == 0:
                self.writer.finish()

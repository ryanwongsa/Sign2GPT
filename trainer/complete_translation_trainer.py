import torch

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
import pickle

from models.get_models import get_model
import copy

from trainer.base.base_trainer import BaseTrainer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from callbacks.full_callback import LoggingCallback
from train_utils.checkpoint_helpers import (
    get_latest_saved_file,
    get_best_checkpoint_details,
)
from collections import defaultdict


class Trainer(BaseTrainer):
    def __init__(self, local_rank, *args, **kwargs):
        super().__init__(args)
        cfg = args[0]
        if "run" in cfg:
            pass
        else:
            self.logger = LoggingCallback(self.cfg)
            self.logger.start_logger()

        if cfg.mixup:
            self.max_length = 128
        else:
            self.max_length = 64

        if "max_length" in cfg:
            self.max_length = cfg.max_length

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
            ["obleu", "orouge"], 
            additional=True,
            max_length=self.max_length,
            num_samples=train_dict["length"],
        )
        self.init_metrics(
            evaluator,
            "valid",
            ["obleu", "orouge"], 
            additional=True,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
        )

        self.init_metrics(
            valid_tester,
            "valid_test",
            [
                "ableu",
                "obleu",
                "orouge",
                "arouge",
            ],  
            additional=False,
            max_length=self.max_length,
            num_samples=valid_dict["length"],
        )
        self.init_metrics(
            test_tester,
            "test_test",
            [
                "ableu",
                "obleu",
                "orouge",
                "arouge",
            ], 
            additional=False,
            max_length=self.max_length,
            num_samples=test_dict["length"],
        )

        def score_function(engine):
            return cfg.score_factor * float(
                engine.state.metrics[cfg.score_name]["bleu4"]
            )

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


        if cfg.save_ckpt:
            def run_tester_valid(engine):
                engine.state.output = None
                engine.state.batch = None
                valid_tester.run(valid_dl, max_epochs=1)

            trainer.add_event_handler(
                Events.EPOCH_COMPLETED(every=10), run_tester_valid
            )
            self.logger.on_valid_epoch_end(trainer, valid_tester)

            if "pbar" in self.cfg.logger_name:
                from ignite.contrib.handlers import ProgressBar

                if idist.get_rank() == 0:
                    pbar = ProgressBar()
                    pbar.attach(valid_tester)

    def dict_metric_from_list(
        self, engine_type, list_of_metrics, dict_metrics, **kwargs
    ):

        if "obleu" in list_of_metrics:
            from metrics.bleu_score import BLEUScore

            def text_transform(x):
                p_words = []
                pred_logits = x["y_pred"]["logits"]
                tgt = x["target"]["targets"]
                for pred, mask in zip(
                    pred_logits.argmax(axis=-1).detach().cpu(),
                    tgt["gt_text_mask"].detach().cpu(),
                ):
                    pred = pred[mask == 1]
                    indices_eos = torch.where(pred == self.tokenizer.eos_token_id)[0]
                    if len(indices_eos) > 0:
                        pred = pred[: indices_eos[0]]
                    p = self.tokenizer.decode(
                        pred,
                        skip_special_tokens=True,
                    )
                    for k, v in self.new_to_original_dict.items():
                        p = p.replace(k, v)

                    p_words.append(
                        " ".join(list(p)) if self.cfg.apply_metric_splitter else p
                    )
                if self.cfg.apply_metric_splitter:
                    t = [" ".join(list(x)) for x in tgt["sentence"]]
                else:
                    t = [x + self.cfg.append_string for x in tgt["sentence"]]
                p_words = [x + self.cfg.append_string for x in p_words]
                # for ind, (pw, tw) in enumerate(zip(p_words, t)):
                #     print(f"{ind} OBLEU:", pw)
                #     print(f"{ind}   TGT:", tw)
                #     print("----------------------------------")
                return p_words, t  # tgt["sentence"]

            dict_metrics[f"{engine_type}/obleu"] = BLEUScore(
                output_transform=text_transform
            )
        if "orouge" in list_of_metrics:
            from metrics.rouge_metric import RougeMetric

            def text_transform(x):
                p_words = []
                pred_logits = x["y_pred"]["logits"]
                tgt = x["target"]["targets"]
                for pred, mask in zip(
                    pred_logits.argmax(axis=-1).detach().cpu(),
                    tgt["gt_text_mask"].detach().cpu(),
                ):
                    pred = pred[mask == 1]
                    indices_eos = torch.where(pred == self.tokenizer.eos_token_id)[0]
                    if len(indices_eos) > 0:
                        pred = pred[: indices_eos[0]]
                    p = self.tokenizer.decode(
                        pred,
                        skip_special_tokens=True,
                    )
                    for k, v in self.new_to_original_dict.items():
                        p = p.replace(k, v)

                    p_words.append(
                        " ".join(list(p)) if self.cfg.apply_metric_splitter else p
                    )
                if self.cfg.apply_metric_splitter:
                    t = [" ".join(list(x)) for x in tgt["sentence"]]
                else:
                    t = [x + self.cfg.append_string for x in tgt["sentence"]]
                p_words = [x + self.cfg.append_string for x in p_words]

                return p_words, t  # tgt["sentence"]

            dict_metrics[f"{engine_type}/orouge"] = RougeMetric(
                output_transform=text_transform
            )
        if "ableu" in list_of_metrics:
            from metrics.bleu_score import BLEUScore

            def autoreg_text_transform(a):
                x = a["y_pred"]["generated"]
                tgt = a["target"]["targets"]
                p_words = []
                for pred in x.detach().cpu():
                    indices_eos = torch.where(pred == self.tokenizer.eos_token_id)[0]
                    if len(indices_eos) > 0:
                        pred = pred[: indices_eos[0]]
                    p = self.tokenizer.decode(
                        pred,
                        skip_special_tokens=True,
                    )
                    for k, v in self.new_to_original_dict.items():
                        p = p.replace(k, v)
                    p_words.append(
                        " ".join(list(p)) if self.cfg.apply_metric_splitter else p
                    )
                if self.cfg.apply_metric_splitter:
                    t = [" ".join(list(x)) for x in tgt["sentence"]]
                else:
                    t = [x + self.cfg.append_string for x in tgt["sentence"]]
                p_words = [x + self.cfg.append_string for x in p_words]
                for ind, (pw, tw) in enumerate(zip(p_words, t)):
                    print(f"{ind} ABLEU:", pw)
                    print(f"{ind}   TGT:", tw)
                    print("----------------------------------")

                return (p_words, t)  # tgt["sentence"])

            dict_metrics[f"{engine_type}/ableu"] = BLEUScore(
                output_transform=autoreg_text_transform
            )
        if "arouge" in list_of_metrics:
            from metrics.bleu_score import BLEUScore

            def autoreg_text_transform2(a):
                x = a["y_pred"]["generated"]
                tgt = a["target"]["targets"]
                p_words = []
                for pred in x.detach().cpu():
                    indices_eos = torch.where(pred == self.tokenizer.eos_token_id)[0]
                    if len(indices_eos) > 0:
                        pred = pred[: indices_eos[0]]
                    p = self.tokenizer.decode(
                        pred,
                        skip_special_tokens=True,
                    )
                    for k, v in self.new_to_original_dict.items():
                        p = p.replace(k, v)

                    p_words.append(
                        " ".join(list(p)) if self.cfg.apply_metric_splitter else p
                    )
                if self.cfg.apply_metric_splitter:
                    t = [" ".join(list(x)) for x in tgt["sentence"]]
                else:
                    t = [x + self.cfg.append_string for x in tgt["sentence"]]
                p_words = [x + self.cfg.append_string for x in p_words]

                return (p_words, t)  # tgt["sentence"])

            dict_metrics[f"{engine_type}/arouge"] = RougeMetric(
                output_transform=autoreg_text_transform2
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.lm_name, **self.cfg.additional_tokens
        )
        if "pretext" in self.cfg and len(self.cfg.pretext) > 0:
            self.pretext = self.cfg.pretext
            self.pretext_tokens = self.tokenizer(self.pretext)["input_ids"]
            self.pretext_length = len(self.pretext_tokens)

            if self.cfg.pretext[-1] == " ":
                self.pretext_tokens = self.pretext_tokens[:-1]
                self.pretext_length = self.pretext_length - 1
        else:
            self.pretext = ""
            self.pretext_tokens = self.tokenizer(self.pretext)["input_ids"]
            self.pretext_length = 1

        dict_model_params = self.cfg.model_params.to_dict()
        dict_model_params["pretext_length"] = self.pretext_length

        if "replacement_pickle" in self.cfg and self.cfg.replacement_pickle is not None:
            with open(self.cfg.replacement_pickle, "rb") as handle:
                dict_replacements = pickle.load(handle)

            self.tokenizer.add_tokens(dict_replacements["new_tokens"])

            dict_model_params["new_token_length"] = len(self.tokenizer)

            self.original_to_new_dict = dict_replacements["original_to_new_dict"]
            self.new_to_original_dict = {
                v: k for k, v in self.original_to_new_dict.items()
            }
        else:
            self.new_to_original_dict = {}

        self.model = get_model(self.cfg.model_name, dict_model_params)
        self.model = idist.auto_model(
            self.model, find_unused_parameters=False, sync_bn=True
        )


    def prep_batch(self, batch, isValid=False, cuda=True):
        idx, frames, sentence = (
            batch["index"],
            batch["frames"],
            batch["sentence"],
        )

        frame_features = frames

        frame_mask = torch.zeros(
            len(frame_features), max([len(feat) for feat in frame_features])
        )
        for i, fr in enumerate(frame_features):
            frame_mask[i, : len(fr)] = 1.0
        frame_mask = frame_mask.bool()

        dict_text = self.tokenizer(
            [
                (self.pretext + sent + self.tokenizer.eos_token)
                # (self.tokenizer.bos_token + sent + self.tokenizer.eos_token)
                for sent in sentence
            ],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        assert torch.all(
            dict_text["input_ids"][:, : self.pretext_length]
            == torch.tensor(self.pretext_tokens).unsqueeze(0)
        )

        text_ids = dict_text["input_ids"][:, :-1]

        text_attention_mask = copy.deepcopy(dict_text["attention_mask"])
        gt_ids = dict_text["input_ids"][:, self.pretext_length :]
        gt_text_mask = text_attention_mask[:, self.pretext_length :].bool()

        res = {
            "model_input": {
                "text_mask": dict_text["attention_mask"][:, :-1].bool(),
                "text_ids": text_ids,
                "frame_features": frame_features,
                "frame_mask": frame_mask,
                "max_len": torch.tensor(
                    self.cfg.max_seq_len if "max_seq_len" in self.cfg else 512
                ),
            },
            "targets": {
                "gloss_ids": batch["gloss_ids"] if "gloss_ids" in batch else [],
                "pseudo_gloss_ids": batch["pseudo_gloss_ids"]
                if "pseudo_gloss_ids" in batch
                else [],
                "index": torch.stack(idx),
                "sentence": sentence,
                "gt_ids": gt_ids,
                "gt_text_mask": gt_text_mask,
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

        with torch.autocast(device_type="cuda", dtype=self.dtype):  # torch.bfloat16):
            self.optimizer.zero_grad()
            y_pred = self.model(**x["model_input"])

        loss, dict_losses = self.loss_fn(y_pred, x["targets"])

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # ------------------------------------------------------------------------
            self.manually_update_gradients()
            # ------------------------------------------------------------------------
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


            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # ------------------------------------------------------------------------
            self.manually_update_gradients()
            # ------------------------------------------------------------------------
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self.model.parameters(), self.grad_clip_value
                )
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

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

                model_input = {
                    "text_mask": x["model_input"]["text_mask"][
                        :, 0 : self.pretext_length
                    ],
                    "text_ids": x["model_input"]["text_ids"][
                        :, 0 : self.pretext_length
                    ],
                    "frame_features": x["model_input"]["frame_features"],
                    "frame_mask": x["model_input"]["frame_mask"],
                    "max_len": x["model_input"]["max_len"],
                }

                generated = self.model(
                    **model_input,
                    gen_params={
                        **self.cfg.gen_params,
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "bos_token_id": self.tokenizer.bos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                    },
                    generate=True,
                )
                y_pred["generated"] = generated["output_ids"]
            loss, dict_losses = self.loss_fn(y_pred, x["targets"])

            return {
                "y_pred": y_pred,
                "target": x,
                "losses": {
                    "loss": loss.detach(),
                    **{k: v.detach() for k, v in dict_losses.items()},
                },
            }

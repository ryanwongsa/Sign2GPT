import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


class Loss(nn.Module):
    def __init__(self, dict_of_loss_params={}):
        super().__init__()
        self.dict_loss_functions = {}
        self.weighting = {}
        dict_mods = {}
        for key, dict_value in dict_of_loss_params.items():

            dict_mods[key] = importlib.import_module(dict_value["cls_name"], package=None)

            self.dict_loss_functions[key] = dict_mods[key].Loss(**dict_value["loss_params"])

            self.weighting[key] = dict_value["weight"]
        
        self.crit_keys = list(dict_of_loss_params.keys())

    def forward(self, y_pred, target):
        loss = 0
        dict_loss = {}
        for key, loss_fn in self.dict_loss_functions.items():
            loss_k = loss_fn(y_pred, target)
            dict_loss[f"loss_{key}"]=loss_k
            loss+=self.weighting[key]*loss_k

        return loss, dict_loss



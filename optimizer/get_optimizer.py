import importlib
import torch
import ignite.distributed as idist
import copy


def get_grouped_params(model, optimizer_params):
    decay = set()
    no_decay = set()

    blacklist_weight_modules = (
        torch.nn.Embedding,
        torch.nn.LayerNorm,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        # torch.nn.Parameter,
    )  # POSITIONAL ENCODING IGNORE WEIGHT DECAY
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if p.requires_grad:
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif not pn.endswith("weight") and isinstance(p, torch.nn.Parameter):
                    no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    union_params = decay | no_decay
    decay = param_dict.keys() - union_params

    params_without_wd = [param_dict[pn] for pn in sorted(list(no_decay))]
    params_with_wd = [param_dict[pn] for pn in sorted(list(decay))]

    optim_p = copy.deepcopy(dict(optimizer_params))
    if "weight_decay" in optim_p:
        wd = optim_p["weight_decay"]
        del optim_p["weight_decay"]
    else:
        wd = 0.0
    list_of_param_groups = [
        {"params": params_with_wd, **optim_p, "weight_decay": wd},
        {"params": params_without_wd, **optim_p, "weight_decay": 0.0},
    ]

    return list_of_param_groups


def get_optim(optimizer_name, optimizer_params, model):
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            get_grouped_params(model, optimizer_params)
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            get_grouped_params(model, optimizer_params)
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            get_grouped_params(model, optimizer_params)
        )
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(
            get_grouped_params(model, optimizer_params)
        )

    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            get_grouped_params(model, optimizer_params)
        )
    optimizer = idist.auto_optim(optimizer)
    print("MODEL TRAINABLE PARAMETERS:")
    for pn, p in model.named_parameters():
        if p.requires_grad:
            print(pn)
    return optimizer

import importlib


def get_loss(criterion_name, criterion_params):
    mod = importlib.import_module(criterion_name, package=None)
    return mod.Loss(**criterion_params)

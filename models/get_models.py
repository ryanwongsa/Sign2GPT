import importlib


def get_model(model_name, model_params):
    mod = importlib.import_module(model_name, package=None)
    return mod.Model(**dict(model_params))  # **model_params.to_dict())

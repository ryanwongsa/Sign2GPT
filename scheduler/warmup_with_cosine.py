from ignite.contrib.handlers import (
    LinearCyclicalScheduler,
    CosineAnnealingScheduler,
    ConcatScheduler,
)
import math
import copy

def get_warmup_with_cosine(args, optimizer, dl):
    lr = args.lr
    if "lr_scale_factor" in args.lr_scheduler_params:
        lr_scale_factor = args.lr_scheduler_params["lr_scale_factor"]
        start_value = lr * lr_scale_factor
    else:
        start_value = args.lr_scheduler_params["start_lr"]
    num_cycles = args.lr_scheduler_params["num_cycles"]
    start_value_mult = args.lr_scheduler_params["start_value_mult"]
    end_value_mult = args.lr_scheduler_params["end_value_mult"]
    warmup_epochs = args.lr_scheduler_params["warmup_epochs"]

    if args.train_length:
        epoch_length = args.train_length
    else:
        epoch_length = len(dl)
    num_epochs = args.max_epochs

    scheduler_1 = LinearCyclicalScheduler(
        optimizer,
        "lr",
        start_value=start_value,
        end_value=lr,
        cycle_size=epoch_length * warmup_epochs * 2,
    )
    scheduler_2 = CosineAnnealingScheduler(
        optimizer,
        "lr",
        start_value=lr,
        end_value=start_value,
        cycle_size=math.ceil(
            ((num_epochs - 1 * warmup_epochs) * epoch_length) / num_cycles
        ),
        start_value_mult=start_value_mult,
        end_value_mult=end_value_mult,
    )
    durations = [
        epoch_length * warmup_epochs,
    ]
    scheduler = ConcatScheduler(
        schedulers=[scheduler_1, scheduler_2], durations=durations
    )
    return scheduler

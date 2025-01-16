from pathlib import Path

import glob


def get_best_checkpoint_details(
    path, extension="pt", best_checkpoint_name="_result_checkpoint_"
):
    list_of_checkpoints = list(Path(path).glob(f"*.{extension}"))
    print(list_of_checkpoints)
    max_score = -100
    for pt in list_of_checkpoints:
        path_cpkt = str(pt).split(best_checkpoint_name)
        if len(path_cpkt) <= 1:
            continue
        path_cpkt = path_cpkt[1].split("_")
        epoch = path_cpkt[0]
        score = float(path_cpkt[-1].split(f".{extension}")[0])
        if max_score < score:
            best_checkpoint = pt
            best_epoch = epoch
            best_score = score
            max_score = score
    try:
        print(best_checkpoint, best_epoch, best_score)
        return str(best_checkpoint), best_epoch, best_score
    except:
        print("NO CHECKPOINT FOUND")
        return "", 0, 0


def get_latest_saved_file(folder, extension="pt", name_latest="latest"):
    list_of_files = list(glob.glob(f"{folder}/*.{extension}"))
    latest_checkpoint = None
    highest_num = 0
    for f in list_of_files:
        if name_latest in Path(f).stem:
            num = int(
                f[f.find(f"{name_latest}_checkpoint") :]
                .split("_")[-1]
                .replace(f".{extension}", "")
            )
            if num >= highest_num:
                latest_checkpoint = f
                highest_num = num
    print(latest_checkpoint)
    return latest_checkpoint, -1, -1


def get_epoch_saved_file(folder, epoch, extension="pt"):
    list_of_files = list(glob.glob(f"{folder}/*.{extension}"))
    latest_checkpoint = None
    highest_num = 0
    for f in list_of_files:
        num = int(f[f.find("checkpoint_") :].split("_")[1])
        if num == epoch:
            latest_checkpoint = f
            highest_num = num
    print(latest_checkpoint)
    return latest_checkpoint, -1, -1

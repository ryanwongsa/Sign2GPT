from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from dataloaders.data_utils.lmdb_utils import LMDBUtility
from dataloaders.data_utils.my_concat_dataset import MyConcatDataset
import ignite.distributed as idist
from tqdm import tqdm


class CSLDailyDataset(Dataset):
    def __init__(
        self,
        id,
        item,
        lmdb_video_dir,
        transform,
        isValid,
        dict_sentence=None,
        dict_lem_to_id=None,
        dict_lem_counter=None,
    ):
        self.id = id
        self.items = [item]
        self.lmdb_video_dir = lmdb_video_dir
        self.transform = transform
        self.isValid = isValid
        self.dict_sentence = dict_sentence
        self.dict_lem_to_id = dict_lem_to_id
        self.dict_lem_counter = dict_lem_counter

        self.lmdb_util_video = None

    def __len__(self):
        return len(self.items)

    def collate_fn(self, batch):
        key_set = {k for k in batch[0].keys()}
        val_list = lambda k: [d.get(k) for d in batch if d.get(k) is not None]

        return {k: val_list(k) for k in key_set}

    def __getitem__(self, idx):
        if not self.lmdb_util_video:
            self.lmdb_util_video = LMDBUtility(
                f"{self.lmdb_video_dir}/{self.id}",
            )
            self.num_frames = self.lmdb_util_video.details["num_frames"]

        item = self.items[idx]
        file_name = item["name"]

        sentence = "".join(item["label_char"])

        if self.isValid:
            start_frame = 0
            end_frame = self.num_frames
        else:
            start_frame = np.random.randint(0, self.transform.random_shift)
            end_frame = np.random.randint(
                self.num_frames - self.transform.random_shift, self.num_frames + 1
            )
        selection_frames = np.arange(start_frame, end_frame, self.transform.stride)
        if len(selection_frames) == 0:
            selection_frames = np.arange(0, self.num_frames)

        frames = self.lmdb_util_video.get_frames(selection_frames)

        if self.transform:
            frames = self.transform.aug_video(frames, self.isValid)
        else:
            frames = torch.tensor(np.stack(frames)).float()

        pseudo_gloss_ids = []
        if self.dict_sentence is not None:
            if sentence in self.dict_sentence:
                lems = self.dict_sentence[sentence]
                pseudo_gloss_ids = [
                    self.dict_lem_to_id[lem]
                    for lem in lems
                    if self.dict_lem_counter[lem] / len(self.dict_sentence) < 0.4
                ]
        return {
            "index": torch.tensor(idx).long(),
            "frames": frames,
            "sentence": sentence,
            "file_name": file_name,
            **(
                {"pseudo_gloss_ids": torch.tensor(pseudo_gloss_ids).long()}
                if self.dict_lem_to_id
                else {}
            ),
        }


def get_ds(ds_params, transform):
    from dataloaders.data_utils.file_utils import read_json, read_pickle

    params = ds_params["ds_params"]
    split = ds_params["split"]
    tsv_dir = ds_params["tsv_dir"]
    df = pd.read_csv(tsv_dir, sep="\t")

    if "pseudo_gloss_dir" in ds_params:
        dict_processed_words = read_pickle(ds_params["pseudo_gloss_dir"])

        dict_sentence = dict_processed_words["dict_sentence"]
        dict_lem_to_id = dict_processed_words["dict_lem_to_id"]
        dict_lem_counter = dict_processed_words["dict_lem_counter"]
    else:
        dict_sentence = None
        dict_lem_to_id = None
        dict_lem_counter = None

    df = df[df.split == split]
    df.label_gloss = df.label_gloss.apply(eval)
    df.label_char = df.label_char.apply(eval)
    df.label_word = df.label_word.apply(eval)
    df.label_postag = df.label_postag.apply(eval)
    list_of_ds = []
    lengths = []
    counter = 0
    for index, row in tqdm(df.iterrows()):
        ds = CSLDailyDataset(
            **params,
            item=row.to_dict(),
            id=row["name"],
            dict_sentence=dict_sentence,
            dict_lem_to_id=dict_lem_to_id,
            dict_lem_counter=dict_lem_counter,
            transform=transform,
        )
        collate_fn = ds.collate_fn
        list_of_ds.append(ds)
        lengths.append(len(ds))
        counter += 1
    ds = MyConcatDataset(list_of_ds)
    dl = idist.auto_dataloader(
        ds,
        shuffle=ds_params["shuffle"],
        sampler=None,
        num_workers=ds_params["num_workers"],
        batch_size=ds_params["bs"],
        drop_last=ds_params["drop_last"],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dl, {"length": np.sum(lengths), "dict_lem_to_id": dict_lem_to_id}


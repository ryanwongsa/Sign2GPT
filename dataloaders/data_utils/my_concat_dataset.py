from torch.utils.data import ConcatDataset
import bisect
import torch

class MyConcatDataset(ConcatDataset):

    def __getitem__(self, idx):

        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        item = self.datasets[dataset_idx][sample_idx]
        item["index"] = torch.tensor(idx)

        return item
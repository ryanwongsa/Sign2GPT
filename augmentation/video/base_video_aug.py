import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from augmentation.video.utils.unnorm import UnNormalize


class Transformation(object):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        strength=1.0,
        random_shift=4,
        stride=1,
        max_seq_len=512,
    ):
        self.random_shift = random_shift
        self.stride = stride
        self.max_seq_len = max_seq_len
        s = strength
        self.transform_train = A.Compose(
            [
                A.Rotate(limit=5, p=0.3),
                A.RandomResizedCrop(
                    height=224,
                    width=224,
                    scale=(0.875, 1),
                    ratio=(0.9, 1.1),
                    p=1.0,
                    always_apply=True,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.8 * s,
                    contrast=0.8 * s,
                    saturation=0.8 * s,
                    hue=0.2 * s,
                    p=0.3,
                ),
                A.ToGray(p=0.2),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    always_apply=True,
                    p=1.0,
                ),
                ToTensorV2(p=1.0, always_apply=True),
            ],
            additional_targets=None,
        )

        self.transform_valid = A.Compose(
            [
                A.Resize(224, 224, always_apply=True, p=1.0),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                    always_apply=True,
                    p=1.0,
                ),
                ToTensorV2(p=1.0, always_apply=True),
            ],
            additional_targets=None,
        )

        self.shift_factor = 0.1
        self.scale_factor = 0.125
        self.unnorm = UnNormalize(mean, std)

    def aug_video(self, frames, isValid):
        image_keys = ["image"]
        additional_targets = {}
        for i in range(1, len(frames)):
            additional_targets[f"image_{i}"] = "image"
            image_keys.append(f"image_{i}")

        sample = {"image": frames[0]}
        for i in range(1, len(frames)):
            sample[f"image_{i}"] = frames[i]

        if isValid:
            self.transform_valid.add_targets(additional_targets)
            sample = self.transform_valid(**sample)
        else:
            self.transform_train.add_targets(additional_targets)
            sample = self.transform_train(**sample)

        return torch.stack([sample[ik] for ik in image_keys])

    def aug_kpt(self, poses, isValid):
        if not isValid:
            shift_values = torch.randn(2) * self.shift_factor
            scale_values = torch.FloatTensor(1).uniform_(
                1.0 - self.scale_factor, 1.0 + self.scale_factor
            )

            for k, pose in poses.items():
                poses[k][:, :, :2] = (pose[:, :, :2] + shift_values) * scale_values

        for k, pose in poses.items():
            poses[k][:, :, :2] = poses[k][:, :, :2] * 2 - 1

        return poses

    def unnorm_imgs(self, frames):
        output = []
        for frame in frames:
            output.append(self.unnorm(frame).detach().cpu().numpy())
        return output

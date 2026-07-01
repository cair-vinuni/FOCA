#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812
from torchvision.transforms.v2 import Compose, ConvertImageDtype



class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    image_tfs: dict[str, ImageTransformConfig] = field(default_factory=dict)
    wrist_tfs: dict[str, ImageTransformConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: dict):
        image_tfs_raw = cfg.get("image_tfs", {})
        image_tfs = {
            k: ImageTransformConfig(**v) if isinstance(v, dict) else v
            for k, v in image_tfs_raw.items()
        }

        wrist_tfs_raw = cfg.get("wrist_tfs", {})
        wrist_tfs = {
            k: ImageTransformConfig(**v) if isinstance(v, dict) else v
            for k, v in wrist_tfs_raw.items()
        }
        return cls(image_tfs=image_tfs, wrist_tfs=wrist_tfs)
    
def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomResizedCrop":
        return v2.RandomResizedCrop(**cfg.kwargs)
    elif cfg.type == "RandomRotate":
        return v2.RandomRotation(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")

class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.image_crop_resize_transform = None
        if "crop_resize" in cfg.image_tfs:
            crop_resize_cfg = cfg.image_tfs["crop_resize"]
            self.image_crop_resize_transform = make_transform_from_config(crop_resize_cfg)

        self.wrist_crop_resize_transform = None
        if "crop_resize" in cfg.wrist_tfs:
            crop_resize_cfg = cfg.wrist_tfs["crop_resize"]
            self.wrist_crop_resize_transform = make_transform_from_config(crop_resize_cfg)


        self.image_transforms = {}
        self.image_weights = []
        for tf_name, tf_cfg in cfg.image_tfs.items():
            if tf_name == "crop_resize" or tf_cfg.weight <= 0.0:
                continue
            self.image_transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.image_weights.append(tf_cfg.weight)

        self.wrist_transforms = {}
        self.wrist_weights = []
        for tf_name, tf_cfg in cfg.wrist_tfs.items():
            if tf_name == "crop_resize" or tf_cfg.weight <= 0.0:
                continue
            self.wrist_transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.wrist_weights.append(tf_cfg.weight)

        n_subset_image = min(len(self.image_transforms), cfg.max_num_transforms)
        n_subset_wrist = min(len(self.wrist_transforms), cfg.max_num_transforms)
        if n_subset_image == 0 or not cfg.enable:
            self.image_tf = v2.Identity()
            self.wrist_tf = v2.Identity()
        else:
            self.image_tf = RandomSubsetApply(
                transforms=list(self.image_transforms.values()),
                p=self.image_weights,
                n_subset=n_subset_image,
                random_order=cfg.random_order,
            )
            self.wrist_tf = RandomSubsetApply(
                transforms=list(self.wrist_transforms.values()),
                p=self.wrist_weights,
                n_subset=n_subset_wrist,
                random_order=cfg.random_order,
            )

            self.image_tf = Compose([
                t for t in [self.image_tf, self.image_crop_resize_transform] if t is not None
            ])
            self.wrist_tf = Compose([
                t for t in [self.wrist_tf, self.wrist_crop_resize_transform] if t is not None
            ])

            # self.tf = Compose([
            #     subset_tf,
            #     ConvertImageDtype(dtype=torch.float32),
            # ])

    def forward(self, *inputs, cam) -> Any:
        if 'wrist' in cam:
            return self.wrist_tf(*inputs)
        else:
            return self.image_tf(*inputs)

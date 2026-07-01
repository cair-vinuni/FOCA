import torch
from lerobot.common.policies.pretrained import PreTrainedPolicy
# from lerobot.common.constants import OBS_STATE, OBS_IMAGE, OBS_WRIST
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

# Add this helper function to normalize a tensor with ImageNet stats
def normalize_imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_STATS["mean"], device=img.device)
    std = torch.tensor(IMAGENET_STATS["std"], device=img.device)
    return (img - mean) / std

class Pi0Inference:
    def __init__(self, model_dir: str, device: str = "cuda"):
        """
        Wrap an existing, loaded PI0Policy for single‐step inference.

        Args:
            policy: an instance of PI0Policy (or any PreTrainedPolicy) with weights already loaded
            device: e.g. "cpu" or "cuda"
        """

        self.device = torch.device(device)
        self.policy = PI0Policy.from_pretrained(model_dir)
        # self.policy.config.n_action_steps = 20
        self.policy = self.policy.to(self.device)
        print(self.policy)
        # self.policy.reset()

    def set_key_for_calvin(self):
        self.image_key = "image"
        self.wrist_image_key = "wrist_image"

    def set_key_for_libero(self):
        self.image_key = "observation.images.image"
        self.wrist_image_key = "observation.images.wrist_image"

    def step(self, data: dict) -> torch.Tensor:
        """
        Run one policy step. 

        Args:
            data = {
                "observation.images.image": ,
                "observation.images.wrist_image":,
                "observation.state":
                "task":,
            }

        Returns:
            action: np.ndarray of shape (action_dim,)
        """

        batch: dict[str, torch.Tensor] = {}

        # state
        state = torch.as_tensor(data["observation.state"], dtype=torch.float32, device=self.device)
        batch["observation.state"] = state.unsqueeze(0)

        # normalize images using ImageNet stats
        for key in (self.image_key, self.wrist_image_key):
            img = torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):  # HWC → CHW
                img = img.permute(2, 0, 1)
            img = img[:3, :, :]  # keep only RGB if 4-channel (e.g. RGBA)
            img = img / 255.0    # scale to [0,1]
            # img = normalize_imagenet(img)
            batch[key] = img.unsqueeze(0)

        batch["task"] = [data["task"]]

        with torch.no_grad():
            action = self.policy.select_action_chunk(batch)
            action = action.cpu().numpy()
        return action[0] # remove batch size dim


import numpy as np
def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
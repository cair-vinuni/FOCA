import os 
import sys 

sys.path.append("/mnt/data/sftp/data/vla_intern/workspace/binh/robocasa_setup/robocasa")
sys.path.insert(0, "/mnt/data/sftp/data/vla_intern/workspace/binh/VLA-Humanoid")

import collections
import dataclasses
import logging
import math
import pathlib
import json

import imageio

import numpy as np
import tqdm
import tyro

import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
from robocasa.utils.eval_utils import create_eval_env
import robocasa.utils.robomimic.robomimic_obs_utils as ObsUtils
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
IMAGENET_STATS = {
    "mean": [[[0.485]], [[0.456]], [[0.406]]],  # (c,1,1)
    "std": [[[0.229]], [[0.224]], [[0.225]]],  # (c,1,1)
}

from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import torch.multiprocessing as mp


ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
DEFAULT_EVAL_UPDATE_KWARGS =  {
    "generative_textures": None,
    "randomize_cameras": False,
    "obj_instance_split": "B",
    "layout_ids": None,
    "style_ids": None,
    "scene_split": None,
    "layout_and_style_ids": [
                    [
                        1,
                        1
                    ],
                    [
                        2,
                        2
                    ],
                    [
                        4,
                        4
                    ],
                    [
                        6,
                        9
                    ],
                    [
                        7,
                        10
                    ]
                ],
    "camera_heights": 256,
    "camera_widths": 256,
}


SHAPE_META = {
    "obs": {
        "robot0_agentview_left_image": {
            "shape": [256, 256,3],
            "type": "rgb"
        },
        "robot0_eye_in_hand_image": {
            "shape": [256, 256, 3],
            "type": "rgb"
        },
        "robot0_agentview_right_image": {
            "shape": [256, 256, 3],
            "type": "rgb"
        },
        "robot0_base_to_eef_pos": {
            "shape": [3]
            # type default: low_dim (not explicitly listed)
        },
        "robot0_base_to_eef_quat": {
            "shape": [4]
        },
        "robot0_gripper_qpos": {
            "shape": [2]
        },
    },
    "action": {
        "shape": [12]
    }
}

# # env_meta_path="/mnt/data/sftp/data/vla_intern/workspace/binh/robocasa_regenerate/notebooks/robocasa_env_meta_5chosen_task.json"
# env_meta_path="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/binh/VLA-Humanoid/robocasa_libs/robocasa/notebooks/robocasa_env_meta_5chosen_task.json"
# with open(env_meta_path, "r") as f:
#     all_task_env_meta = json.load(f) 



class Pi0ClientPolicy:

    def __init__(self, model_dir=""):
        self.model_dir = model_dir
        self.device = 'cuda'
        self.client = self.create_client()

    def get_robocasa_action(self, obs_dict, lang, chunk=10):
        data = self._process_observation(obs_dict, lang)
        
        batch: dict[str, torch.Tensor] = {}
        
        state = torch.as_tensor(data["state"], dtype=torch.float32, device=self.device)
        batch["state"] = state.unsqueeze(0)
        # print("State:", batch["state"])
        # normalize images using ImageNet stats
        for key in ("right_image", "left_image", "wrist_image"):
            img = torch.as_tensor(data[key], dtype=torch.float32, device=self.device)
            if img.ndim == 3 and img.shape[2] in (1, 3, 4):  # HWC → CHW
                img = img.permute(2, 0, 1)
            img = img[:3, :, :]  # keep only RGB if 4-channel (e.g. RGBA)
            img = img / 255.0    # scale to [0,1]
            batch[key] = img.unsqueeze(0)
        batch["task"] = [data["task"]]
        
        with torch.no_grad():
            action = self.client.select_action_chunk(batch)
            action = action.cpu().numpy()
        action = action[0]
        action = self.normalize_gripper_action(action)
        return action[:10]   

    def create_client(self):
        
        try:
            policy = PI0Policy.from_pretrained(self.model_dir)
            policy = policy.to("cuda")           
            return policy
        except Exception as e:
            print(f"❌ Failed to load policy: {e}")
            sys.exit(1)
        

    def _process_observation(self, obs, lang):

        left_img = obs["robot0_agentview_left_image"][::-1, :, :]
        right_img = obs["robot0_agentview_right_image"][::-1, :, :]
        wrist_img = obs["robot0_eye_in_hand_image"][::-1, :, :] 
        
        left_img = np.ascontiguousarray(left_img)
        right_img = np.ascontiguousarray(right_img)
        wrist_img = np.ascontiguousarray(wrist_img)

        # import pdb
        # pdb.set_trace()
        new_obs = {
            "left_image": left_img, 
            "right_image": right_img, 
            "wrist_image": wrist_img, 
            "state": np.concatenate(
                            (
                                obs["robot0_base_to_eef_pos"],
                                obs["robot0_base_to_eef_quat"],
                                obs["robot0_gripper_qpos"],
                            )
                        ), 
            "task": lang, 
            
        }
        
        return new_obs
    

    def normalize_gripper_action(self, action, binarize=True):
        orig_low, orig_high = 0.0, 1.0
        action[..., 6] = 2 * (action[..., 6] - orig_low) / (orig_high - orig_low) - 1

        if binarize:
            # Binarize to -1 or +1.
            action[..., 6] = np.sign(action[..., 6])

        # action[..., 7:] = 0
        # action[..., -1] = -1
        return action


@dataclasses.dataclass
class Args:

    replan_steps: int = 10
    model_dir: str="/mnt/data/sftp/data/vla_intern/workspace/binh/VLA-Humanoid/outputs/train/2026-01-10/00-17-14_pi0_base_robocasa_100demos_24tasks_gradacc2/checkpoints/040000/pretrained_model"
    # model_dir: str="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/nghiem/VLA-Humanoid_merge/outputs/train/2026-01-02/07-18-46_robocasa_100_scales_original_batch12/checkpoints/070000/pretrained_model"
    env_name: str = "TurnOnMicrowave"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials: int = 5 #50  # Number of rollouts per task
    horizon: int = 800  # Number of steps to run in each episode
    video_out_path: str = f"data/robocasa/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    exp_name: str = "test"
    gpu: int = 0


def eval_robocasa(args: Args, task:str=None) -> None:
    
    # set up logger
    log_dir = pathlib.Path(f"robocasa_eval_logs/{args.exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    if task is not None:
        args.env_name=task
        
    args.video_out_path = f'{args.video_out_path}/{args.exp_name}/{args.env_name}'

    # 2) clear any existing handlers and install a FileHandler
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_file = log_dir / f"{args.env_name}_gpu{args.gpu}.log"
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    )
    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    
    # Set random seed
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = Pi0ClientPolicy(args.model_dir)
    logging.info(f"Load model from: {args.model_dir}")

    # Start evaluation
    total_episodes, total_successes = 0, 0
    # Get task
    env = create_eval_env(args.env_name, seed=args.seed,
                           camera_widths=256, camera_heights=256)
    # Start episodes
    task_episodes, task_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials)):

        # Reset environment
        env.reset()
        task_lang = env.get_ep_meta()['lang']
        # action_plan = collections.deque()
        
        # Setup
        t = 0
        # replay_images = []
        
        replay_images_wrist = []
        replay_images_left = []
        replay_images_right = []

        logging.info(f"Starting episode {task_episodes+1}...| Task description: {task_lang}")
        while t < args.horizon + args.num_steps_wait:

            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(ROBOCASA_DUMMY_ACTION)
                t += 1
                continue
            
            action_chunk = client.get_robocasa_action(obs, task_lang)
            # print(action_chunk)
            # Execute action in environment
            # obs, reward, done, info = env.step(action_chunk.tolist())

            for act in action_chunk:
                obs, reward, done, info = env.step(act.tolist())
                done = done or env._check_success()
                t +=1
            
                # replay_img = obs["robot0_eye_in_hand_image"][::-1, :, :]   # or env.render(...), either is fine
                # replay_images.append(to_video_frame(replay_img))

                replay_img_wrist = obs["robot0_eye_in_hand_image"][::-1, :, :]  
                replay_images_wrist.append(to_video_frame(replay_img_wrist))
            
                replay_img_left = obs["robot0_agentview_left_image"][::-1, :, :]  
                replay_images_left.append(to_video_frame(replay_img_left))
                
                replay_img_right = obs["robot0_agentview_right_image"][::-1, :, :]  
                replay_images_right.append(to_video_frame(replay_img_right))
            
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            if done:
                break

        task_episodes += 1
        total_episodes += 1

        # Save a replay video of the episode
        suffix = "success" if done else "failure"
        # imageio.mimwrite(
        #     pathlib.Path(args.video_out_path) / f"rollout_{task_lang}_{suffix}.mp4",
        #     [np.asarray(x) for x in replay_images],
        #     fps=10,
        # )
        # ================================================================================
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_seed_{args.seed}_trial_{episode_idx}_wrist_{task_lang}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images_wrist],
            fps=10,
        )
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_seed_{args.seed}_trial_{episode_idx}_left_{task_lang}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images_left],
            fps=10,
        )
        imageio.mimwrite(
            pathlib.Path(args.video_out_path) / f"rollout_seed_{args.seed}_trial_{episode_idx}_right_{task_lang}_{suffix}.mp4",
            [np.asarray(x) for x in replay_images_right],
            fps=10,
        )
        # ================================================================================

        # Log current results
        logging.info(f"Success: {done}")
        logging.info(f"# episodes completed so far: {total_episodes}")
        logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")



import numpy as np

def to_video_frame(arr):
    arr = np.asarray(arr)
    if any(s < 0 for s in arr.strides) or not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))  
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 3 and arr.shape[-1] > 4:
        arr = arr[..., :3]

    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        if arr.shape[-1] == 2:
            arr = arr[..., 0] 
        else:
            raise ValueError(f"Unexpected channel count: {arr.shape}")

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if 0.0 <= vmin and vmax <= 1.0:
            arr = (arr * 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).round().astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    arr = np.ascontiguousarray(arr)
    return arr


def eval_robocasa_all(args:Args):
    print("=" * 80)
    print("🎯 ROBOCASA Simulation Evaluation")
    print("=" * 80)
    tasks = [
        'PnPCabToCounter',
        'PnPCounterToCab',
        'CoffeeSetupMug',
        'TurnOffStove',
        'TurnOnMicrowave'
    ]
    ctx = mp.get_context("spawn")
    results = dict()
    
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futures = {pool.submit(eval_robocasa, args, task): task for task in tasks}
        for fut in as_completed(futures):
            task = futures[fut]
            try:
                results[task] = fut.result()
            except Exception as e:
                print(f"[ERROR] Task '{task}' failed: {e}")

    print("All done. Results:", results)

if __name__ == "__main__":
    
    # Set spawn before any CUDA/PyTorch code runs
    try:
        mp.set_start_method("spawn", force=True)
        # If you use torch.multiprocessing anywhere:
        import torch
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    tyro.cli(eval_robocasa_all)

    
    # logging.basicConfig(level=logging.INFO)
    # tyro.cli(eval_robocasa)
    
    
'''


python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_40k_24tasks_gradacc2 \
   --args.num-trials 25 \
  --args.seed 0 \
   --args.gpu 0

python scripts/eval_robocasa.py \
   --args.exp-name base_robocasa_100demos_40k_24tasks_gradacc2 \
   --args.num-trials 25 \
  --args.seed 1 \
   --args.gpu 1

'''

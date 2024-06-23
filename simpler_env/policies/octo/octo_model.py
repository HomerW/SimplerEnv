from collections import deque
from typing import Optional, Sequence
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
from octo.model.octo_model import OctoModel
import tensorflow as tf
from transforms3d.euler import euler2axangle

from simpler_env.utils.action.action_ensemble import ActionEnsembler

import logging
from pathlib import Path


def download_checkpoint_from_gcs(cloud_path: str, step: str, save_path: str):
    if not cloud_path.startswith("gs://"):
        return cloud_path, step  # Actually on the local filesystem

    checkpoint_path = tf.io.gfile.join(cloud_path, f"{step}")
    ds_stats_path = tf.io.gfile.join(cloud_path, "dataset_statistics*")
    config_path = tf.io.gfile.join(cloud_path, "config.json*")
    example_batch_path = tf.io.gfile.join(cloud_path, "example_batch.msgpack*")

    run_name = Path(cloud_path).name
    save_path = os.path.join(save_path, run_name)

    target_checkpoint_path = os.path.join(save_path, f"{step}")
    if os.path.exists(target_checkpoint_path):
        logging.warning("Checkpoint already exists at %s, skipping download", target_checkpoint_path)
        return save_path, step
    os.makedirs(save_path, exist_ok=True)
    logging.warning("Downloading checkpoint and metadata to %s", save_path)

    os.system(f"gsutil cp -r {checkpoint_path} {save_path}/")
    os.system(f"gsutil cp {ds_stats_path} {save_path}/")
    os.system(f"gsutil cp {config_path} {save_path}/")
    os.system(f"gsutil cp {example_batch_path} {save_path}/")

    return save_path, step


class OctoInference:
    def __init__(
        self,
        model_type: str = "octo-base",
        step: int = 300000,
        checkpoint_cache_dir: str = "/mnt2/homer/checkpoints",
        policy_setup: str = "widowx_bridge",
        horizon: int = 2,
        pred_action_horizon: int = 4,
        image_size: int = 256,
        head_name: str = "action",
        action_scale: float = 1.0,
        init_rng: int = 0,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            self.dataset_id = "bridge_dataset"
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            self.dataset_id = "fractal20220817_data"
            action_ensemble = True
            action_ensemble_temp = 0.0
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported for octo models.")
        self.policy_setup = policy_setup

        if model_type in ["octo-base", "octo-small", "octo-base-1.5", "octo-small-1.5"]:
            # released huggingface octo models
            self.model_type = f"hf://rail-berkeley/{model_type}"
            step=None
        else:
            # custom model path
            weights_path, step = download_checkpoint_from_gcs(model_type, step, checkpoint_cache_dir)
            self.model_type = weights_path

        self.model = OctoModel.load_pretrained(self.model_type, step=step)
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp
        self.rng = jax.random.PRNGKey(init_rng)
        self.head_name = head_name

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.gripper_is_closed = False

        self.task = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(self.pred_action_horizon, self.action_ensemble_temp)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = tf.image.resize(
            image,
            size=(self.image_size, self.image_size),
            method="lanczos3",
            antialias=True,
        )
        image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        # self.image_history.append(image)
        # Alternative implementation below; but looks like for real eval, filling the entire buffer at the first step is not necessary
        if self.num_image_history == 0:
            self.image_history.extend([image] * self.horizon)
        else:
            self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _obtain_image_history_and_mask(self) -> tuple[np.ndarray, np.ndarray]:
        images = np.stack(self.image_history, axis=0)
        horizon = len(self.image_history)
        timestep_pad_mask = np.ones(horizon, dtype=np.float64)  # note: this should be of float type, not a bool type
        timestep_pad_mask[: horizon - min(horizon, self.num_image_history)] = 0
        # timestep_pad_mask[: horizon - min(2, self.num_image_history)] = 0
        return images, timestep_pad_mask

    def reset(self, task_description: str) -> None:
        self.task = self.model.create_tasks(texts=[task_description])
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.gripper_is_closed = False

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                # task description has changed; reset the policy state
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images, timestep_pad_mask = self._obtain_image_history_and_mask()
        images, timestep_pad_mask = images[None], timestep_pad_mask[None]

        # we need use a different rng key for each model forward step; this has a large impact on model performance
        self.rng, key = jax.random.split(self.rng)  # each shape [2,]
        # print("octo local rng", self.rng, key)

        input_observation = {"image_primary": images, "timestep_pad_mask": timestep_pad_mask}
        raw_actions = self.model.sample_actions(
            input_observation,
            self.task,
            unnormalization_statistics=self.model.dataset_statistics[self.dataset_id]["action"],
            head_name=self.head_name,
            rng=key,
        )
        raw_actions = raw_actions[0]  # remove batch, becoming (action_pred_horizon, action_dim)
        assert raw_actions.shape == (self.pred_action_horizon, 7)

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
            raw_actions = raw_actions[None]  # [1, 7]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            # This is one of the ways to implement gripper actions; we use an alternative implementation below for consistency with real
            # gripper_close_commanded = (current_gripper_action < 0.5)
            # relative_gripper_action = 1 if gripper_close_commanded else -1 # google robot 1 = close; -1 = open

            # # if action represents a change in gripper state and gripper is not already sticky, trigger sticky gripper
            # if gripper_close_commanded != self.gripper_is_closed and not self.sticky_action_is_on:
            #     self.sticky_action_is_on = True
            #     self.sticky_gripper_action = relative_gripper_action

            # if self.sticky_action_is_on:
            #     self.gripper_action_repeat += 1
            #     relative_gripper_action = self.sticky_gripper_action

            # if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
            #     self.gripper_is_closed = (self.sticky_gripper_action > 0)
            #     self.sticky_action_is_on = False
            #     self.gripper_action_repeat = 0

            # action['gripper'] = np.array([relative_gripper_action])

            # alternative implementation
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = (
                    self.previous_gripper_action - current_gripper_action
                )  # google robot 1 = close; -1 = open
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and self.sticky_action_is_on is False:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = (
                2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
            )  # binarize gripper action to 1 (open) and -1 (close)
            # self.gripper_is_closed = (action['gripper'] < 0.0)

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

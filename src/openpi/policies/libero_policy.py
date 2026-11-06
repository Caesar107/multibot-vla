import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared import camera_geometry

# Keys carrying the base ("agentview") camera geometry, used for Plücker camera conditioning.
# `cam_K` (9,) is the row-major 3x3 pinhole matrix *at the resolution of the accompanying observation/image*
# (256x256 in the LeRobot dataset, 224x224 from the eval client); `cam_c2w` (16,) is the row-major 4x4 MuJoCo
# camera-to-world matrix (camera looks down -z). Both describe the raw render; the 180 degree rotation that
# LIBERO images undergo is applied to the ray map here.
CAM_K_KEY = "observation/cam_K"
CAM_C2W_KEY = "observation/cam_c2w"


def make_libero_example(*, with_camera: bool = False) -> dict:
    """Creates a random input example for the Libero policy."""
    example = {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }
    if with_camera:
        example[CAM_K_KEY] = camera_geometry.intrinsics_from_fovy(45.0, 224, 224).reshape(-1)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 3] = [0.66, 0.0, 1.61]
        example[CAM_C2W_KEY] = c2w.reshape(-1)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    # Image keys for which a Plücker ray map is produced from `observation/cam_K` / `observation/cam_c2w`
    # (camera conditioning, see `Pi0Config.plucker_image_keys`). Only "base_0_rgb" is supported for LIBERO.
    plucker_image_keys: tuple[str, ...] = ()
    # Resolution of the ray map; must match the model's image resolution.
    plucker_resolution: tuple[int, int] = _model.IMAGE_RESOLUTION

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if self.plucker_image_keys:
            if set(self.plucker_image_keys) != {"base_0_rgb"}:
                raise ValueError(f"LIBERO only has camera parameters for base_0_rgb, got {self.plucker_image_keys}")
            if CAM_K_KEY not in data or CAM_C2W_KEY not in data:
                # Original single-camera LIBERO has no per-frame extrinsics; fill in the canonical agentview.
                cam_k, cam_c2w = camera_geometry.canonical_libero_camera(*base_image.shape[:2])
                data = {**data, CAM_K_KEY: cam_k, CAM_C2W_KEY: cam_c2w}
            inputs["plucker"] = {
                "base_0_rgb": camera_geometry.plucker_from_flat(
                    data[CAM_K_KEY],
                    data[CAM_C2W_KEY],
                    src_hw=tuple(base_image.shape[:2]),
                    dst_hw=tuple(self.plucker_resolution),
                    rotate_180=True,
                )
            }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][..., :7])}

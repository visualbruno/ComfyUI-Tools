import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from os.path import join
import glob
import numpy as np
import torch
import trimesh
import json
import cv2
import pointops
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from transformers import pipeline, SamModel
from PIL import Image

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .sampart3d_util import *


@DATASETS.register_module()
class SAMPart3DDataset16Views(Dataset):

    def __init__(
        self,
        split="train",
        data_root="data/scannet",
        mesh_root="",
        mesh_path_mapping=None,
        oid="",
        label="",
        sample_num=15000,
        pixels_per_image=256,
        batch_size=90,
        transform=None,
        loop=1,
        extent_scale=10.0
    ):
        super(SAMPart3DDataset16Views, self).__init__()

        data_root = os.path.join(data_root, str(oid))
        mesh_path = os.path.join(mesh_root, f"{oid}.glb")
        self.data_root = data_root
        self.split = split
        self.pixels_per_image = pixels_per_image
        self.batch_size = batch_size
        self.device = 'cuda'
        self.logger = get_root_logger()

        self.extent_scale = extent_scale

        self.meta_data = json.load(open(os.path.join(data_root, "meta.json")))

        # Load mesh and sample pointclouds
        self.mesh_path = mesh_path
        transform = Compose(transform)
        self.load_mesh(mesh_path, transform, sample_num)

        # Prepare SAM masks and depth mapping
        if self.split == "train":

            self.prepare_meta_data()

            self.loop = loop
            self.data_list = self.get_data_list()
            self.logger.info(
                "Totally {} x {} samples in {} set.".format(
                    len(self.data_list), self.loop, split
                )
            )

    def sample_pixel(self, masks, image_height=512, image_width=512):
        masks = masks.to(self.device)
        indices_batch = torch.zeros((self.batch_size*self.pixels_per_image, 3), device=self.device)
        random_imgs = torch.randint(0, len(masks), (self.batch_size,), device=self.device)
        for i in range(self.batch_size):
            # Find the indices of the valid points in the mask
            valid_indices = torch.nonzero(masks[random_imgs[i]], as_tuple=False)
            # if len(valid_indices) == 0:
            #     continue
            # Randomly sample from the valid indices
            if len(valid_indices) >= self.pixels_per_image:
                indices = valid_indices[torch.randint(0, len(valid_indices), (self.pixels_per_image,))]
            else:
                # Repeat the indices to fill up to pixels_per_image
                repeat_times = self.pixels_per_image // len(valid_indices) + 1
                indices = valid_indices.repeat(repeat_times, 1)[:self.pixels_per_image]

            indices_batch[i * self.pixels_per_image : (i + 1) * self.pixels_per_image, 0] = random_imgs[i]
            indices_batch[i * self.pixels_per_image : (i + 1) * self.pixels_per_image, 1:] = indices

        return indices_batch


    def load_mesh(self, mesh_path, transform, sample_num=15000, pcd_path=None):
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        coord, face_index, color = sample_surface(mesh, count=sample_num, sample_color=True)
        color = color[..., :3]
        face_normals = mesh.face_normals
        normal = face_normals[face_index]
        # self.mesh_scale, self.mesh_center_offset = cal_scale(mesh_path)
        mesh_scale = self.meta_data["scaling_factor"]
        mesh_center_offset = self.meta_data["mesh_offset"]

        object_org_coord = coord.copy()
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]])
        object_org_coord = np.dot(object_org_coord, rotation_matrix)
        object_org_coord = object_org_coord * mesh_scale + mesh_center_offset

        offset = torch.tensor(coord.shape[0])
        obj = dict(coord=coord, normal=normal, color=color, offset=offset, origin_coord=object_org_coord, face_index=face_index)
        obj = transform(obj)
        self.object_org_coord = obj["origin_coord"].clone()
        self.face_index = obj["face_index"].clone().numpy()
        self.pcd_inverse = obj["inverse"].clone().numpy()
        # print("object_org_coord", torch.unique(self.object_org_coord, return_counts=True))
        del obj["origin_coord"], obj["face_index"], obj["inverse"]
        self.object = obj
        
            

    def prepare_meta_data(self, data_path=None):
        SAM_model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
        pixel_level_keys_list = []
        scale_list = []
        group_cdf_list = []
        depth_valid_list = []
        mapping_list = []
        mapping_valid_list = []
        object_org_coord = self.object_org_coord.to(self.device).contiguous().float()
        obj_offset = torch.tensor(object_org_coord.shape[0]).to(self.device)

        camera_angle_x = self.meta_data['camera_angle_x']
        for i, c2w_opengl in enumerate(self.meta_data["transforms"]):
            # print(frame['index'])
            c2w_opengl = np.array(c2w_opengl)
            self.logger.info(f"Processing frame_{i}")
            rgb_path = join(self.data_root, f"render_{i:04d}.webp")
            img = np.array(Image.open(rgb_path))
            if img.shape[-1] == 4:
                mask_img = img[..., 3] == 0
                img[mask_img] = [255, 255, 255, 255]
                img = img[..., :3]
                img = Image.fromarray(img.astype('uint8'))

            # Calculate mapping
            depth_path = join(self.data_root, f"depth_{i:04d}.exr")
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = depth[..., 0]
            depth_valid = torch.tensor(depth < 65500.0)
            org_points = gen_pcd(depth, c2w_opengl, camera_angle_x)
            org_points = torch.from_numpy(org_points)
            points_tensor = org_points.to(self.device).contiguous().float()
            offset = torch.tensor(points_tensor.shape[0]).to(self.device)
            indices, distances = pointops.knn_query(1, object_org_coord, obj_offset, points_tensor, offset)
            mapping = torch.zeros((depth.shape[0], depth.shape[1]), dtype=torch.int) - 1

            # Create a mask where distances are less than 0.03
            mask_dis = distances[..., 0] < 0.03
            indices[~mask_dis] = -1
            mapping[depth_valid] = indices.cpu().flatten()
            mapping_valid = mapping != -1

            # Calculate groups 
            try:
                masks = SAM_model(img, points_per_side=32, pred_iou_thresh=0.9, stability_score_thresh=0.9)
                masks = masks['masks']
                masks = sorted(masks, key=lambda x: x.sum())
            except:
                masks = []

            # mask filter
            masks_filtered = []
            img_valid = ~mask_img
            for mask in masks:
                valid_ratio = mask[img_valid].sum() / img_valid.sum()
                invalid_ratio = mask[mask_img].sum() / mask_img.sum()
                if valid_ratio == 0 or invalid_ratio > 0.1:
                    continue
                else:
                    masks_filtered.append(mask)
            pixel_level_keys, scale, mask_cdf = self._calculate_3d_groups(torch.from_numpy(depth), mapping_valid, masks_filtered, points_tensor[mask_dis])    

            pixel_level_keys_list.append(pixel_level_keys)
            scale_list.append(scale)
            group_cdf_list.append(mask_cdf)
            depth_valid_list.append(depth_valid)
            mapping_list.append(mapping)
            mapping_valid_list.append(mapping_valid)

        self.pixel_level_keys = torch.nested.nested_tensor(
        pixel_level_keys_list
        )
        self.scale_3d_statistics = torch.cat(scale_list)
        self.scale_3d = torch.nested.nested_tensor(scale_list)
        self.group_cdf = torch.nested.nested_tensor(group_cdf_list)
        self.depth_valid = torch.stack(depth_valid_list)
        self.mapping = torch.stack(mapping_list)
        self.mapping_valid = torch.stack(mapping_valid_list)

    def _calculate_3d_groups(
        self,
        depth: torch.Tensor,
        valid: torch.Tensor,
        masks: torch.Tensor,
        point: torch.Tensor,
        max_scale: float = 2.0,
    ):
        """
        Calculate the set of groups and their 3D scale for each pixel, and the cdf.
        Returns:
            - pixel_level_keys: [H, W, max_masks]
            - scale: [num_masks, 1]
            - mask_cdf: [H, W, max_masks]
        max_masks is the maximum number of masks that was assigned to a pixel in the image,
         padded with -1s. mask_cdf does *not* include the -1s.
        Refer to the main paper for more details.
        """
        image_shape = depth.shape[:2]
        depth_valid = valid
        point = point.to(self.device)

        def helper_return_no_masks():
            # Fail gracefully when no masks are found.
            # Create dummy data (all -1s), which will be ignored later.
            # See: `get_loss_dict_group` in `garfield_model.py`
            pixel_level_keys = torch.full(
                (image_shape[0], image_shape[1], 1), -1, dtype=torch.int
            )
            scale = torch.Tensor([0.0]).view(-1, 1)
            mask_cdf = torch.full(
                (image_shape[0], image_shape[1], 1), 1, dtype=torch.float
            )
            return (pixel_level_keys, scale, mask_cdf)


        # If no masks are found, return dummy data.
        if len(masks) == 0:
            return helper_return_no_masks()

        sam_mask = []
        scale = []

        # For all 2D groups,
        # 1) Denoise the masks (through eroding)
        all_masks = torch.stack(
            # [torch.from_numpy(_["segmentation"]).to(self.device) for _ in masks]
            [torch.from_numpy(_).to(self.device) for _ in masks]
        )
        # erode all masks using 3x3 kernel
        # ignore erode
        eroded_masks = torch.conv2d(
            all_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        # 2) Calculate 3D scale
        # Don't include groups with scale > max_scale (likely to be too noisy to be useful)
        for i in range(len(masks)):
            curr_mask_org = eroded_masks[i]
            curr_mask = curr_mask_org[depth_valid]
            curr_points = point[curr_mask]
            extent = (curr_points.std(dim=0) * self.extent_scale).norm()
            if extent.item() < max_scale:
                sam_mask.append(curr_mask_org)
                scale.append(extent.item())

        # If no masks are found, after postprocessing, return dummy data.
        if len(sam_mask) == 0:
            return helper_return_no_masks()

        sam_mask = torch.stack(sam_mask)  # (num_masks, H, W)
        scale = torch.Tensor(scale).view(-1, 1).to(self.device)  # (num_masks, 1)

        # Calculate "pixel level keys", which is a 2D array of shape (H, W, max_masks)
        # Each pixel has a list of group indices that it belongs to, in order of increasing scale.
        pixel_level_keys = self.create_pixel_mask_array(
            sam_mask
        ).long()  # (H, W, max_masks)
        depth_invalid = ~depth_valid
        pixel_level_keys[depth_invalid, :] = -1

        # Calculate group sampling CDF, to bias sampling towards smaller groups
        # Be careful to not include -1s in the CDF (padding, or unlabeled pixels)
        # Inversely proportional to log of mask size.
        mask_inds, counts = torch.unique(pixel_level_keys, return_counts=True)
        counts[0] = 0  # don't include -1
        probs = counts / counts.sum()  # [-1, 0, ...]

        pixel_shape = pixel_level_keys.shape
        if (pixel_level_keys.max()+2) != probs.shape[0]:
            pixel_level_keys_new = pixel_level_keys.reshape(-1)
            unique_values, inverse_indices = torch.unique(pixel_level_keys_new, return_inverse=True)
            pixel_level_keys_new = inverse_indices.reshape(-1)
        else:
            pixel_level_keys_new = pixel_level_keys.reshape(-1) + 1

        mask_probs = torch.gather(probs, 0, pixel_level_keys.reshape(-1) + 1).view(
            pixel_shape
        )
        mask_log_probs = torch.log(mask_probs)
        never_masked = mask_log_probs.isinf()
        mask_log_probs[never_masked] = 0.0
        mask_log_probs = mask_log_probs / (
            mask_log_probs.sum(dim=-1, keepdim=True) + 1e-6
        )
        mask_cdf = torch.cumsum(mask_log_probs, dim=-1)
        mask_cdf[never_masked] = 1.0

        return (pixel_level_keys.cpu(), scale.cpu(), mask_cdf.cpu())
    
    @staticmethod
    def create_pixel_mask_array(masks: torch.Tensor):
        """
        Create per-pixel data structure for grouping supervision.
        pixel_mask_array[x, y] = [m1, m2, ...] means that pixel (x, y) belongs to masks m1, m2, ...
        where Area(m1) < Area(m2) < ... (sorted by area).
        """
        max_masks = masks.sum(dim=0).max().item()
        # print(max_masks)
        image_shape = masks.shape[1:]
        pixel_mask_array = torch.full(
            (max_masks, image_shape[0], image_shape[1]), -1, dtype=torch.int
        ).to(masks.device)

        for m, mask in enumerate(masks):
            mask_clone = mask.clone()
            for i in range(max_masks):
                free = pixel_mask_array[i] == -1
                masked_area = mask_clone == 1
                right_index = free & masked_area
                if len(pixel_mask_array[i][right_index]) != 0:
                    pixel_mask_array[i][right_index] = m
                mask_clone[right_index] = 0
        pixel_mask_array = pixel_mask_array.permute(1, 2, 0)

        return pixel_mask_array

    def get_data_list(self):
        data_list = glob.glob(os.path.join(self.data_root, "*.exr"))
        return data_list

    def get_data(self, idx):
        indices = self.sample_pixel(self.mapping_valid, 512, 512).long().detach().cpu()
        npximg = self.pixels_per_image
        img_ind = indices[:, 0]
        x_ind = indices[:, 1]
        y_ind = indices[:, 2]

        # sampled_imgs = img_ind[::npximg]
        mask_id = torch.zeros((indices.shape[0],), device=self.device)
        scale = torch.zeros((indices.shape[0],), device=self.device)
        mapping = torch.zeros((indices.shape[0],), device=self.device)

        random_vec_sampling = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)
        random_vec_densify = (torch.rand((1,)) * torch.ones((npximg,))).view(-1, 1)

        for i in range(0, indices.shape[0], npximg):
            img_idx = img_ind[i]

            # calculate mapping
            mapping[i : i + npximg] = self.mapping[img_idx][x_ind[i : i + npximg], y_ind[i : i + npximg]]

            # Use `random_vec` to choose a group for each pixel.
            per_pixel_index = self.pixel_level_keys[img_idx][
                x_ind[i : i + npximg], y_ind[i : i + npximg]
            ]
            random_index = torch.sum(
                random_vec_sampling.view(-1, 1)
                > self.group_cdf[img_idx][x_ind[i : i + npximg], y_ind[i : i + npximg]],
                dim=-1,
            )

            # `per_pixel_index` encodes the list of groups that each pixel belongs to.
            # If there's only one group, then `per_pixel_index` is a 1D tensor
            # -- this will mess up the future `gather` operations.
            if per_pixel_index.shape[-1] == 1:
                per_pixel_mask = per_pixel_index.squeeze()
            else:
                per_pixel_mask = torch.gather(
                    per_pixel_index, 1, random_index.unsqueeze(-1)
                ).squeeze()
                per_pixel_mask_ = torch.gather(
                    per_pixel_index,
                    1,
                    torch.max(random_index.unsqueeze(-1) - 1, torch.Tensor([0]).int()),
                ).squeeze()

            mask_id[i : i + npximg] = per_pixel_mask.to(self.device)

            # interval scale supervision
            curr_scale = self.scale_3d[img_idx][per_pixel_mask]
            curr_scale[random_index == 0] = (
                self.scale_3d[img_idx][per_pixel_mask][random_index == 0]
                * random_vec_densify[random_index == 0]
            )
            for j in range(1, self.group_cdf[img_idx].shape[-1]):
                if (random_index == j).sum() == 0:
                    continue
                curr_scale[random_index == j] = (
                    self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    + (
                        self.scale_3d[img_idx][per_pixel_mask][random_index == j]
                        - self.scale_3d[img_idx][per_pixel_mask_][random_index == j]
                    )
                    * random_vec_densify[random_index == j]
                )
            scale[i : i + npximg] = curr_scale.squeeze().to(self.device)
        
        batch = dict()
        batch["mask_id"] = mask_id
        batch["scale"] = scale
        batch["nPxImg"] = npximg
        batch["obj"] = self.object
        batch["mapping"] = mapping.long()
        return batch

    def val_data(self):
        return dict(obj=self.object)

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def __getitem__(self, idx):
        return self.get_data(idx % len(self.data_list))

    def __len__(self):
        return len(self.data_list) * self.loop

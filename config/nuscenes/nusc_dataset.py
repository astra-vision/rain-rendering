import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from torchvision.datasets import VisionDataset


class NuScenesDataset(VisionDataset):
    def __init__(self, version, root, transform=None, target_transform=None, *, verbose=True,
                 specific_tokens=None, sensor_modality='camera', sensor='CAM_FRONT', lidar='LIDAR_TOP',
                 pretransform_data=True, preload_data=True, only_annotated=False):
        super(NuScenesDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.nusc = NuScenes(version=version, dataroot=root, verbose=verbose)
        self.lidar = lidar
        self.only_annotated = only_annotated
        self.sensor = ""
        self.sensor_modality = ""
        if specific_tokens:
            self.tokens = specific_tokens
        elif sensor:
            self.tokens = self.nusc.field2token(table_name="sample_data", field="channel",
                                                query=sensor)
            self.sensor = sensor
        elif sensor_modality:
            self.tokens = self.nusc.field2token(table_name="sample_data", field="sensor_modality",
                                                query=sensor_modality)
            self.sensor_modality = sensor_modality
        else:
            raise ValueError("Both sensor_modality or sensor parameters cannot be None.")

        if only_annotated:
            tokens = []
            for t in self.tokens:
                sample_data = self.nusc.get("sample_data", t)
                if sample_data["is_key_frame"]:
                    tokens.append(t)
            self.tokens = tokens
        if verbose:
            print("Number of valid sample data tokens: {}".format(len(self.tokens)))

        self.objects = []
        self.images = []
        self.scene_tokens = []
        self.transforms = transforms
        self.pretransform_data = pretransform_data
        self.preload_data = preload_data
        if self.preload_data:
            for t in self.tokens:
                img = Image.open(self.get_filepath(t))
                if self.transform and self.pretransform_data:
                    img = self.transform(img)
                self.images.append(img)
                self.objects = []
        for t in self.tokens:
            # find scene token
            self.scene_tokens.append(self.get_scene_token(t))

    def __getitem__(self, index):
        if self.preload_data:
            img = self.images[index]
        else:
            img = Image.open(self.get_filepath(self.tokens[index])).convert('RGB')

        if self.transform and not self.pretransform_data:
            img = self.transform(img)

        # TODO:: return object detection groundtruth
        return img, self.tokens[index]

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return self.nusc.__repr__()

    def get_filepath(self, token):
        assert token in self.tokens, "Token {} not in specific tokens set".format(token)
        sample_data = self.nusc.get("sample_data", token)
        return os.path.join(self.nusc.dataroot, sample_data["filename"])

    def get_filepaths(self, scene_token, sensor="", use_specific_tokens=True):
        sensor = sensor if sensor else self.sensor
        assert sensor

        scene = self.nusc.get("scene", scene_token)
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        first_sample_data_token = first_sample["data"][sensor]

        curr_token = first_sample_data_token
        file_paths = []
        all_tokens = set(self.tokens)  # Checking if curr_token is in the token list would be slower
        while curr_token:
            sample_data = self.nusc.get("sample_data", curr_token)
            curr_token = sample_data["next"]

            if use_specific_tokens and sample_data["token"] not in all_tokens:
                continue

            if not self.only_annotated or sample_data["is_key_frame"]:
                file_paths.append(sample_data["filename"])
        return file_paths

    def get_scene_token(self, sample_data_token):
        sample_data = self.nusc.get("sample_data", sample_data_token)
        sample = self.nusc.get("sample", sample_data["sample_token"])
        return sample["scene_token"]

    def estimate_camera_settings(self, sensor=""):
        # Returns 1 camera settings per scene (dict, scene token)
        sensor = sensor if sensor else self.sensor
        assert sensor

        camera_settings = dict()
        for t in set(self.scene_tokens):
            scene = self.nusc.get("scene", t)
            sample = self.nusc.get("sample", scene["first_sample_token"])

            first_sample_data_token = sample["data"][sensor]

            sample_data = self.nusc.get("sample_data", first_sample_data_token)
            calibrated_sensor = self.nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

            # needs ccd parameters (since exif is a myth ;) ) took from https://www.nuscenes.org/data-collection
            ccd_width = 1600
            ccd_height = 1200
            px_size = 1.98  # in um --> more or less arbitrary... considering CMOS 1/8'' with 1600 px width
            exposure = 20  # in ms

            # calculate "focal" from ccd parameters and intrinsics matrix
            intrinsics = np.array(calibrated_sensor["camera_intrinsic"])
            assert np.any(intrinsics)  # this could fail, if, for any reason, I would be reading radar sensor intrinsics
            focal = np.mean([intrinsics[0, 0] * px_size / 1000, intrinsics[1, 1] * px_size / 1000])  # focal in mm

            # Hardcoded known fact because laziness
            frequency = 12
            width = 1600
            height = 900

            camera_settings[t] = ({"translation": calibrated_sensor["translation"], "focal": focal, "px_size": px_size,
                                   "ccd_width": ccd_width, "ccd_height": ccd_height, "width": width, "height": height,
                                   "frequency": frequency, "exposure": exposure })

        return camera_settings

    def estimate_camera_motions(self, sensor=""):
        # Returns a list of camera motion per scene (dict, scene token)
        sensor = sensor if sensor else self.sensor
        assert sensor

        camera_motions = dict()
        for t in set(self.scene_tokens):
            scene = self.nusc.get("scene", t)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            first_sample_data_token = first_sample["data"][sensor]

            scene_motions = []
            curr_token = first_sample_data_token
            last_camera_position = None
            while curr_token:
                sample_data = self.nusc.get("sample_data", curr_token)
                curr_token = sample_data["next"]
                ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

                if sample_data["token"] == first_sample_data_token:
                    last_camera_position = np.array(ego_pose["translation"])
                    continue

                scene_motions.append((np.array(ego_pose["translation"]) - last_camera_position).tolist())
                last_camera_position = ego_pose["translation"]
            scene_motions.append(scene_motions[-1])  # Yep, last frame will keep its momentum
            camera_motions[t] = scene_motions

        return camera_motions

    def estimate_sequences_duration(self, sensor="", epsilon=1e-3):
        # Returns 1 duration per scene (dict, scene token)
        sensor = sensor if sensor else self.sensor
        assert sensor

        # Hardcoded known fact because laziness
        frequency = 12

        scenes_duration = dict()
        for t in set(self.scene_tokens):
            scene = self.nusc.get("scene", t)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            first_sample_data_token = first_sample["data"][sensor]

            curr_token = first_sample_data_token
            tokens = []
            while curr_token:
                tokens.append(curr_token)
                sample_data = self.nusc.get("sample_data", curr_token)
                curr_token = sample_data["next"]

            scenes_duration[t] = len(tokens) / frequency + epsilon  # duration in sec

        return scenes_duration

    def get_depth_from_lidar(self, sample_data_token):
        sample_token = self.nusc.get("sample_data", sample_data_token)["sample_token"]
        sample = self.nusc.get("sample", sample_token)
        sample_data_lidar_token = sample["data"][self.lidar]

        pts_cloud, depths = self.map_pointcloud_to_image(sample_data_lidar_token, sample_data_token)
        pts_cloud[2, :] = depths

        return pts_cloud

    def map_pointcloud_to_image(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
        plane. [Recoded from the NuscenesExplorer class so the image is not to be loaded].
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>).
        """

        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Retrieve the color from the depth.
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < 1600 - 1)    # hardcoded width
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < 900 - 1)   # hardcoded height
        points = points[:, mask]
        coloring = coloring[mask]

        return points, coloring


class NuScenesGANDataset(NuScenesDataset):
    def __init__(self, version, root, gan_root, post_fix="", transform=None, target_transform=None, *, verbose=True,
                 specific_tokens=None, sensor_modality='camera', sensor='CAM_FRONT', lidar='LIDAR_TOP',
                 pretransform_data=True, preload_data=True, only_annotated=False):
        super(NuScenesGANDataset, self).__init__(version, root, transform, target_transform, verbose=verbose,
                                                 specific_tokens=specific_tokens, sensor_modality=sensor_modality,
                                                 sensor=sensor, lidar=lidar, pretransform_data=pretransform_data,
                                                 preload_data=preload_data, only_annotated=only_annotated)
        self.gan_root = gan_root
        self.post_fix = post_fix

    def get_filepath(self, token):
        assert token in self.tokens, "Token {} not in specific tokens set".format(token)
        sample_data = self.nusc.get("sample_data", token)
        f = os.path.splitext(os.path.basename(sample_data["filename"]))
        # hardcoded to png
        return os.path.join(self.gan_root, f[0] + self.post_fix + ".png")

    def get_filepaths(self, scene_token, sensor="", use_specific_tokens=True):
        sensor = sensor if sensor else self.sensor
        assert sensor

        scene = self.nusc.get("scene", scene_token)
        first_sample = self.nusc.get("sample", scene["first_sample_token"])
        first_sample_data_token = first_sample["data"][sensor]

        curr_token = first_sample_data_token
        file_paths = []
        all_tokens = set(self.tokens)  # Checking if curr_token is in the token list would be slower
        while curr_token:
            sample_data = self.nusc.get("sample_data", curr_token)
            curr_token = sample_data["next"]

            if use_specific_tokens and sample_data["token"] not in all_tokens:
                continue

            if not self.only_annotated or sample_data["is_key_frame"]:
                f = sample_data["filename"]

                # HARDCODED IN PNG since we will never save results in a lossy format
                file_paths.append(os.path.splitext(f)[0] + ".png")
        return file_paths


if __name__ == "__main__":
    dataset = NuScenesDataset(version='v1.0-mini', root=os.path.join('../../data.nobkp', 'nuscenes'),
                              pretransform_data=False, preload_data=False, only_annotated=True,
                              transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))[0]
    fixed_imgs = np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0))
    plt.figure()
    plt.imshow(fixed_imgs)
    plt.axis("off")
    plt.show()

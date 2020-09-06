import numpy as np

from common import my_utils

########################
'''
IMPORTANT: World position of drops are with respect to camera coordinates where (0,0,0) is the camera position. 
The direction of Y-axis is opposite to that taken in the KITTI dataset.
'''


class DropDepthMap:
	def __init__(self, filename):
		self.filename = filename
		self.P2_R_rect = None
		self.P2_R_inv = None
		self.camera_pos_world = None
		self.world_pts_acc_cam = None
		self.camera_pos_wrt_cam0 = None

	def get_util_matrices(self):
		with open(self.filename, 'r') as file:
			lines = file.read().split('\n')
		for line in lines:
			if line[0:10]=='P_rect_02:':
				P2_rect = np.array(line.split(':')[1].split(' ')[1:]).astype(float).reshape((3, 4))

			elif line[0:10] == 'R_rect_02:':
				R2_rect = np.array(line.split(':')[1].split(' ')[1:]).astype(float).reshape((3, 3))

		# Making R2 4x4 for enabling dot product with P2 (4x4)
		R2_rect_44 = np.identity(4).astype(float)
		R2_rect_44[:3, :3] = R2_rect

		# Ground is taken to be (0,0,0). height at which camera0 is situated is 1.65m
		self.world_pts_acc_cam = np.array([0., 1.65, 0.0]).reshape((3, 1))

		# Position of camera wrt origin coord
		camera_pos_wrt_world = - self.world_pts_acc_cam

		# Position of cam2 with reference to cam0.
		# This is done because cam2 is shifted a little in x-direction wrt origin
		self.camera_pos_wrt_cam0 = np.zeros((3, 1))
		self.camera_pos_wrt_cam0[0] = P2_rect[0, 3]/(-P2_rect[0, 0])

		# Final position of cam2 wrt origin
		self.camera_pos_world = self.camera_pos_wrt_cam0 + camera_pos_wrt_world

		# Inverse calib matrix calculation
		self.P2_R_rect = np.dot(P2_rect, R2_rect_44)
		self.P2_R_inv = np.linalg.pinv(self.P2_R_rect)

	def return_xyz(self, depth_map):

		x = np.arange(depth_map.shape[1])
		y = np.arange(depth_map.shape[0])
		z = np.ones((depth_map.shape[0], depth_map.shape[1], 1))

		xx, yy = np.meshgrid(x, y)

		xx = np.expand_dims(xx, axis=-1)
		yy = np.expand_dims(yy, axis=-1)

		xyz_image = np.concatenate((xx, yy, z), axis=-1)
		xyz_coord = np.dot(self.P2_R_inv, xyz_image.reshape((-1, 3)).T).T
		xyz_coord = np.reshape(xyz_coord, (352, 1216, 4))
		xyz_coord = xyz_coord[:, :, :3]
		scale_term = depth_map/xyz_coord[:, :, 2]
		scale_term = np.expand_dims(scale_term, axis=-1)
		xyz_coord *= scale_term

		return xyz_coord

	def get_world_points(self,depth_map):

		self.get_util_matrices()

		xyz_coord = self.return_xyz(depth_map)

		xyz_coord[:, :, 1] = -xyz_coord[:, :, 1]

		return xyz_coord

	@staticmethod
	def depth_map_drop(drops_start, xyz_map):
		depth_maps = np.zeros((drops_start.shape[0], xyz_map.shape[0], xyz_map.shape[1])).astype(np.float16)
		xyz_maps = np.tile(np.expand_dims(xyz_map, 0), (drops_start.shape[0], 1, 1, 1))
		drops_start = np.reshape(drops_start, (-1, 1, 1, 3))
		depth_maps[:, :, :] = np.sqrt(np.square(xyz_maps[:, :, :, 0] - drops_start[:, :, :, 0]) + np.square(
			xyz_maps[:, :, :, 1] - drops_start[:, :, :, 1]) + np.square(xyz_maps[:, :, :, 2] - drops_start[:, :, :, 2]))
		my_utils.print_success('Depth Maps Generated')
		return depth_maps

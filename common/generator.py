import os
import sys
import time

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from natsort import natsorted

from common import add_attenuation, my_utils
from common import solid_angle
from common.bad_weather import DBManager, DropType, RainRenderer, EnvironmentMapGenerator, FovComputation
from common.drop_depth_map import DropDepthMap

plt.ion()

FOG_ATT = 1
USE_DEPTH_WEIGHTING = 0  # TODO: not used for a while

class Generator:
    def __init__(self, args):
        # strategy
        self.conflict_strategy = args.conflict_strategy
        self.rendering_strategy = args.rendering_strategy

        # output paths
        if args.rendering_strategy is None:
            self.output_root = os.path.join(args.output, args.dataset)
        else:
            self.output_root = os.path.join(args.output, args.dataset + '_' + args.rendering_strategy)

        # dataset info
        self.dataset = args.dataset
        self.dataset_root = args.dataset_root
        self.images = args.images
        self.sequences = args.sequences
        self.depth = args.depth
        self.particles = args.particles
        self.weather = args.weather
        self.texture = args.texture
        self.norm_coeff = args.norm_coeff
        self.save_envmap = args.save_envmap
        self.settings = args.settings

        # dataset specific
        self.calib = args.calib

        # camera info
        self.exposure = args.settings["cam_exposure"]
        self.camera_gain = args.settings["cam_gain"]
        self.focal = args.settings["cam_focal"] / 1000.
        self.f_number = args.settings["cam_f_number"]
        self.focus_plane = args.settings["cam_focus_plane"]

        # aesthetic params
        self.noise_scale = args.noise_scale
        self.noise_std = args.noise_std
        self.opacity_attenuation = args.opacity_attenuation

        # generator run params
        self.frame_start = args.frame_start
        self.frame_end = args.frame_end
        self.frame_step = args.frame_step
        self.frames = args.frames
        self.verbose = args.verbose

        # options for environment map and irradiance types
        self.env_type = 'ours'  # 'pano' | 'ours'
        self.irrad_type = 'ambient'  # 'garg' | 'ambient'


        # initialize to None internal big frame by frame object
        self.db = None
        self.renderer = None
        self.fov_comp = None
        self.BGR_env_map = None
        self.env_map_xyY = None
        self.solid_angle_map = None

        # check if everything is fine
        self.check_folders()

    def check_folders(self):
        print('Output directory: {}'.format(self.output_root))

        # Verify existing folders
        existing_folders = []
        for sequence in self.sequences:
            for w in self.weather:
                # loading simulator file path
                out_dir = os.path.join(self.output_root, sequence, w["weather"], '{}mm'.format(w["fallrate"]))

                if os.path.exists(out_dir):
                    existing_folders.append(out_dir)

        if len(existing_folders) != 0 and self.conflict_strategy is None:
            print("\r\nFolders already exist: \n%s" % "\n".join([d for d in existing_folders]))
            while self.conflict_strategy not in ["overwrite", "skip", "rename_folder"]:
                self.conflict_strategy = input(
                    "\r\nWhat strategy to use (overwrite|skip|rename_folder):   ")

        assert(self.conflict_strategy in [None, "overwrite", "skip", "rename_folder"])

    @staticmethod
    def crop_drop(streak):
        streak = (streak * 255).astype(np.uint8)
        im = Image.fromarray(streak)
        background = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, background)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bBox = diff.getbbox()

        im_cropped = im.crop(bBox)
        im_cropped = np.asarray(im_cropped) / 255
        return im_cropped.astype(np.float64)

    def compute_drop(self, bg, drop_dict, rainy_bg, rainy_mask, rainy_saturation_mask):
        # Drop taken from database
        streak_db_drop = self.db.take_drop_texture(drop_dict)

        image_height, image_width = bg.shape[:2]

        # Gaussian streaks do not need a perspective warping. If strak is not BIG -> Gaussian streak
        if drop_dict.drop_type == DropType.Big:
            pts1, pts2, maxC, minC = self.renderer.warping_points(drop_dict, streak_db_drop, image_width, image_height)
            shape = np.subtract(maxC, minC).astype(int)
            perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            drop = cv2.warpPerspective(streak_db_drop, perspective_matrix, (max(shape[0], 1), max(shape[1], 1)),
                                       flags=cv2.INTER_CUBIC)
            drop = np.clip(drop, 0, 1)
        else:
            # in case of drops from database
            # Gaussian noise to simulate soft wind (in degrees)
            noise = np.random.normal(0.0, self.noise_std) * self.noise_scale

            dir1 = drop_dict.image_position_start - drop_dict.image_position_end
            n1 = np.linalg.norm(dir1)
            dir1 = dir1 / n1
            dir2 = np.array([0, -1])

            # Drop angle in degrees; add small random gaussian noise to represent localized wind
            theta = np.rad2deg(np.arccos(np.dot(dir1, dir2)))

            # Note: The noise is added to the drop coordinates AFTER the drop angle is calculated so the rotate_bound
            # function, which uses interpolation (contrarily to the drop position which are in integers),
            # would be more accurate
            nx, ny = np.cos(np.deg2rad(noise)), np.sin(np.deg2rad(noise))
            mean_x = (drop_dict.image_position_end[0] + drop_dict.image_position_start[0]) / 2
            mean_y = (drop_dict.image_position_end[1] + drop_dict.image_position_start[1]) / 2
            drop_dict.image_position_start[:] = \
                (drop_dict.image_position_start[0] - mean_x) * nx - \
                (drop_dict.image_position_start[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_start[0] - mean_x) * ny + \
                (drop_dict.image_position_start[1] - mean_y) * nx + mean_y
            drop_dict.image_position_end[:] = \
                (drop_dict.image_position_end[0] - mean_x) * nx - \
                (drop_dict.image_position_end[1] - mean_y) * ny + mean_x,\
                (drop_dict.image_position_end[0] - mean_x) * ny + \
                (drop_dict.image_position_end[1] - mean_y) * nx + mean_y

            drop = imutils.rotate_bound(streak_db_drop, theta + noise)

            drop = cv2.flip(drop, 0) if drop_dict.image_position_end[0] > rainy_bg.shape[1] // 2 else drop
            height = max(abs(drop_dict.image_position_end[1] - drop_dict.image_position_start[1]), 2)
            width = max(abs(
                drop_dict.image_position_end[0] - drop_dict.image_position_start[0]), drop_dict.max_width + 2)
            drop = cv2.resize(drop, (width, height), interpolation=cv2.INTER_AREA)
            drop = np.clip(drop, 0, 1)
            minC = drop_dict.image_position_start

        # Compute alpha channel from any other channel (since it was gray)
        drop = np.dstack([drop, drop[..., 0]])

        ###########################################   COLOUR DROP  #################################################
        drop_fov_pts, drop_fov_pts3d, drop_direction, drop_position = \
            self.fov_comp.compute_fov_plane_points(drop_dict, self.renderer.radius, self.renderer.fov,
                                                   20, self.BGR_env_map.shape)
        try:
            rainy_bg, rainy_mask, rainy_saturation_mask, drop, blended_drop, minC = \
                self.renderer.add_drop_to_image(self.dataset, self.env_map_xyY, self.solid_angle_map, drop_fov_pts,
                                                minC, bg, rainy_bg, rainy_mask, rainy_saturation_mask, drop, drop_dict,
                                                self.irrad_type, self.rendering_strategy, self.opacity_attenuation)
        except Exception as e:
            import traceback
            print('Erroneous drop (' + str(e) + ')')
            print(traceback.print_exc())
            blended_drop = None

        return rainy_bg, rainy_mask, rainy_saturation_mask, drop, blended_drop, minC

    def run(self):
        process_t0 = time.time()

        folders_num = len(self.images)

        # case for any number of sequences and supported rain intensities
        for folder_idx, sequence in enumerate(self.sequences):
            folder_t0 = time.time()
            print('\nSequence: ' + sequence)
            sim_num = len(self.particles[sequence])
            depth_folder = self.depth[sequence]

            for sim_idx, sim_weather in enumerate(self.weather):
                weather, fallrate = sim_weather["weather"], sim_weather["fallrate"]

                out_seq_dir = os.path.join(self.output_root, sequence)
                out_dir = os.path.join(out_seq_dir, weather, '{}mm'.format(fallrate))
                sim_file = self.particles[sequence][sim_idx]

                # Resolve output path
                path_exists = os.path.exists(out_dir)
                if path_exists:
                    if self.conflict_strategy == "skip":
                        pass
                    elif self.conflict_strategy == "overwrite":
                        pass
                    elif self.conflict_strategy == "rename_folder":
                        out_dir_, out_shift = out_dir, 0
                        while os.path.exists(out_dir_ + '_copy%05d' % out_shift):
                            out_shift += 1

                        out_dir = out_dir_ + '_copy%05d' % out_shift
                    else:
                        raise NotImplementedError

                # Create directory
                os.makedirs(out_dir, exist_ok=True)

                # Default fog-like rain parameters
                fog_params = {"rain_intensity": fallrate, "focal": self.focal, "f_number": self.f_number, "angle": 90,
                              "exposure": self.exposure, "camera_gain": self.camera_gain}

                if "nuscenes" in self.dataset:
                    files = self.images[sequence]
                    depth_files = self.depth[sequence]
                    depth_file = depth_files[0]
                    assert depth_file.endswith(".npy"), "nuscenes processing only works with .npy for depth"
                    # depth = np.load(depth_file)
                    if "gan" in self.dataset:
                        # HARDCODED since these values are not known in nuscenes_gan
                        imW, imH = (1600, 900)
                    else:
                        img = cv2.imread(files[0])
                        imH, imW = img.shape[0:2]
                else:
                    files = natsorted(np.array([os.path.join(self.images[sequence], picture) for picture in my_utils.os_listdir(self.images[sequence])]))
                    depth_files = natsorted(np.array([os.path.join(depth_folder, depth) for depth in my_utils.os_listdir(depth_folder)]))
                    im = files[0]
                    if im.endswith(".png"):
                        imH, imW = cv2.imread(im).shape[0:2]
                    elif im.endswith(".npy"):
                        imH, imW = np.load(im).shape[0:2]
                    else:
                        raise Exception("Invalid extension", im)
                    imH = imH//self.settings["render_scale"]
                    imW = imW//self.settings["render_scale"]

                if self.camera_gain:
                    fog_params["camera_gain"] = self.camera_gain

                print('Simulation: rain {}mm/hr'.format(fallrate))
                # loading StreaksDBManager, RainRenderer, FOVComputation and EnvMapGenerator
                self.db = DBManager(streaks_path_xml=sim_file, streaks_path=self.texture,
                                    norm_coeff_path=self.norm_coeff)
                self.renderer = RainRenderer(focal=self.focal, f_number=self.f_number, focus_plane=6, radius=10, fov=165)
                self.fov_comp = FovComputation(camera=np.array([0, 0, 0]))
                map_generator = EnvironmentMapGenerator(self.focal, imW, imH)

                # loading fog class
                FOG = add_attenuation.FogRain(**fog_params)

                # loading calib files for frames
                calib_files = self.calib[sequence]

                # loading streaks from Streaks Database
                self.db.load_streak_database()

                # creating drops based on the simulator file
                self.db.load_streaks_from_xml(self.dataset, self.settings, [imW, imH], use_pickle=False, verbose=self.verbose)

                env_map_input = self.env_type if self.irrad_type == 'garg' else self.env_type + '_' + self.irrad_type

                frame_render_dict = list(self.db.streaks_simulator.values())

                f_start, f_end, f_step = self.frame_start, self.frame_end, self.frame_step
                f_end = len(files) if f_end is None else min(f_end, len(files))
                if self.frames:
                    # prone to go "boom", so we clip and remove 'wrong' ids
                    idx = np.unique(np.clip(self.frames, 0, f_end - 1)).tolist()
                else:
                    idx = list(range(f_start, f_end, f_step))  # to make it

                f_num = len(idx)
                sim_t0 = time.time()
                print("{} images".format(len(idx)))
                frames_exist_nb = 0
                for f_idx, i in enumerate(idx):
                    image_file = files[i]
                    depth_file = depth_files[i]

                    # Compute frame index (independent of starting frame) to allow deterministic / reproducible rendering
                    if self.dataset == 'nuscenes':
                        # It could be useful for other dataset, but, for the moment, lets consign this little gem of a
                        # code to nuscenes
                        # If f_end was not supplied, we could see if the number of file is not equal to the number of
                        # simulated drop... if so, lets remap them
                        render_ix = np.linspace(0, len(frame_render_dict), len(files), endpoint=False, dtype=int)
                        f_name_idx = render_ix[i]
                    else:
                        f_name_idx = i

                    assert os.path.exists(image_file), "Image file {} does not exist".format(image_file)
                    assert os.path.exists(depth_file), "Depth file {} does not exist".format(depth_file)

                    # Ensure deterministic behavior
                    np.random.seed(f_name_idx)

                    frame_t0 = time.time()
                    frame = frame_render_dict[f_name_idx % len(frame_render_dict)]
                    file_name = os.path.split(image_file)[-1]

                    out_rainy_path = os.path.join(out_dir, 'rainy_image', '{}.png'.format(file_name[:-4]))
                    out_rainy_mask_path = os.path.join(out_dir, 'rain_mask', '{}.png'.format(file_name[:-4]))
                    out_env_path = os.path.join(out_seq_dir, 'envmap', '{}.png'.format(file_name[:-4]))

                    frame_exists = os.path.exists(out_rainy_path) or os.path.exists(out_rainy_mask_path)
                    if frame_exists:
                        if self.conflict_strategy == "skip":
                            frames_exist_nb += 1
                            continue
                        elif self.conflict_strategy == "overwrite":
                            pass
                        else:
                            raise NotImplementedError

                    # TODO adding functions of depth weighting for other dataset
                    if USE_DEPTH_WEIGHTING == 1 and calib_files is not None:
                        # Compute the drop depth map to allow weighting envmap from drop FOV
                        depth_drop_evaluator = DropDepthMap(filename=calib_files[0])

                    # flush should happens after a while
                    if self.verbose:
                        sys.stdout.write(
                            '\r' + my_utils.process_eta_str(process_t0, folder_idx, folders_num, folder_t0, sim_idx,
                                                            sim_num, sim_t0, f_idx,
                                                            f_num, frame_t0) + '                        ')

                    # two copies of bg because one is used for rain drop calculation
                    # and the other is changed after adding each drop
                    bg = cv2.imread(image_file) / 255.0

                    if self.settings["render_scale"] != 1:
                        bg = cv2.resize(bg, (int(bg.shape[1]//self.settings["render_scale"]), int(bg.shape[0]//self.settings["render_scale"])))

                    if FOG_ATT == 1:
                        # Depth map is used in weighting the luminance effect of the environment map on a single drop
                        if depth_file.endswith(".png"):
                            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
                            if depth is None:
                                print('Missing/Corrupted depth data (%s)' % depth_file)
                                continue

                            depth = depth.astype(np.float32) / 256.
                        elif depth_file.endswith(".npy"):
                            depth = np.load(depth_file)
                        else:
                            raise Exception("Invalid extension")

                        # Apply depth and render scale
                        depthHW = np.array([int((depth.shape[0] * self.settings["depth_scale"]) // self.settings["render_scale"]), int((depth.shape[1] * self.settings["depth_scale"]) // self.settings["render_scale"])])
                        if not np.all(depth.shape[:2] == depthHW):
                            depth = cv2.resize(depth, (depthHW[1], depthHW[0]))

                        assert (np.all(depth.shape[:2] <= bg.shape[:2])), "Depth cannot be larger than the image"

                        # Strategy to apply if RGB and Depth size mismatch
                        if not np.all(depth.shape[:2] == bg.shape[:2]):
                            # print("\nDepth {} size ({},{}) differs from image ({},{}). Will assume depth is crop centered.".format(image_file, depth.shape[0], depth.shape[1], bg.shape[0], bg.shape[1]))
                            bg = my_utils.crop_center(bg, depth.shape[0], depth.shape[1])
                    else:
                        # In that case, no need for depth, but it's still used down in the code, for more less nothing
                        depth = np.zeros((bg.shape[1], bg.shape[0]), np.float)

                    rainy_bg = FOG.fog_rain_layer(bg, depth)

                    # rain layer is the image of the rendered rain drops blended with the background
                    rain_layer = np.zeros((bg.shape[0], bg.shape[1], 4), np.float64)

                    # rainy_mask keeps track of the pixels of the background that have already been occupied by rain.
                    # This is used in cases of occluding or partially occluded drops
                    rainy_mask = np.zeros((bg.shape[0], bg.shape[1]), np.float64)
                    rainy_saturation_mask = np.zeros((bg.shape[0], bg.shape[1], 3), np.float64)

                    # Environment map of the frame using (Christopher Cameron, 2005):
                    # http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f05/pub/www/projects/fproj/cmcamero/report.pdf
                    if 'ours' in env_map_input:
                        # print('\nGenerating environment map')
                        self.BGR_env_map = map_generator.generate_map(rainy_bg)
                    elif 'pano' in env_map_input:
                        # print('\nLoading Environment Pano')
                        self.BGR_env_map = cv2.imread(os.path.join('../data', 'panos', file_name)) / 255.0
                    else:
                        raise NotImplementedError

                    self.env_map_xyY = my_utils.convert_rgb_to_xyY(self.BGR_env_map[..., ::-1])
                    self.env_map_xyY[np.isnan(self.env_map_xyY)] = 0

                    self.solid_angle_map = solid_angle.get_solid_angles(self.BGR_env_map)

                    # Render only streaks inside the frame
                    streak_dict = frame.streaks
                    streak_dict = {keys: values for keys, values in streak_dict.items() if
                                   1 <= values.max_width < max(imH, imW) and
                                   1 <= values.length < max(imH, imW) and
                                   ((0 <= values.image_position_start[0] < imW
                                     and 0 <= values.image_position_start[1] < imH) or
                                    (0 <= values.image_position_end[0] < imW
                                     and 0 <= values.image_position_end[1] < imH))}

                    if USE_DEPTH_WEIGHTING == 1:
                        xyz_coord = depth_drop_evaluator.get_world_points(depth)

                    assert streak_dict.__len__() <= 2 ** 16, \
                        "Assert that the number of drops doesn't overpass the uint16 rain_mask capacity"

                    streak_list = list(streak_dict.values())
                    drop_num = len(streak_list)
                    drop_process_t0 = time.time()
                    for drop_idx, drop_dict in enumerate(streak_list):
                        # Returns the rainy image, rainy_mask, drop, blended drop and the starting coord of
                        # the drop in image
                        rainy_bg, rainy_mask, rainy_saturation_mask, \
                        drop, blended_drop, minC = self.compute_drop(bg, drop_dict, rainy_bg,
                                                                     rainy_mask, rainy_saturation_mask)
                        if blended_drop is not None:
                            rain_layer = self.renderer.make_rain_layer(drop, blended_drop, rain_layer, rainy_mask, minC)
                        else:
                            print("Trace: rain drop {} in sequence {} in image {} ({})".format(drop_idx,
                                                                                               sequence, f_idx,
                                                                                               f_name_idx))

                        # Compute progress
                        avg_drop_time = (time.time() - drop_process_t0) / (drop_idx + 1)
                        if self.verbose or drop_idx == 0:
                            sys.stdout.write(
                                '\r' + my_utils.process_eta_str(process_t0, folder_idx, folders_num, folder_t0, sim_idx,
                                                                sim_num,
                                                                sim_t0, f_idx, f_num, frame_t0, drop_idx,
                                                                drop_num) + '\t\t' + '%.1fms /drop' % (
                                        1000. * avg_drop_time) + '       ')

                    # Create all output directories
                    os.makedirs(os.path.dirname(out_rainy_path), exist_ok=True)
                    os.makedirs(os.path.dirname(out_rainy_mask_path), exist_ok=True)
                    if self.save_envmap:
                        os.makedirs(os.path.dirname(out_env_path), exist_ok=True)

                    # mean contrast adjusted image
                    rainy_bg_mean = np.mean(rainy_bg)
                    bg_mean = np.mean(bg)
                    difference_mean = rainy_bg_mean - bg_mean
                    rainy_bg_copy = rainy_bg - difference_mean

                    plt.imsave(out_rainy_path, np.clip(rainy_bg_copy[..., ::-1], 0, 1))
                    plt.imsave(out_rainy_mask_path, rainy_mask)
                    if self.save_envmap:
                        plt.imsave(out_env_path, self.BGR_env_map[..., ::-1])
                if frames_exist_nb > 0:
                    print("Skipped {}/{} already existing renderings".format(frames_exist_nb, len(idx)))

            print("\n\nEnd of the simulation")








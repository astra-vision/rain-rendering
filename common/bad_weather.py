import hashlib
import os
import sys
import traceback
from enum import Enum
from xml.etree.ElementTree import parse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyclipper
from scipy.ndimage.filters import gaussian_filter

from common import my_utils, db

plt.ion()

try:
    import cPickle as pickle
except Exception:
    import pickle


'''
###########################################
#       IMPORTANT                         #
###########################################                                        

- The positive direction of the Z-Axis is being reversed in our code as compared to the one in simulation file
- The positive direction of the Y-Axis is being reversed in our code as compared to the KITTI dataset

## Conventions (comments about papers)
# vr - Vision and Rain (Garg & Nayar 2007)
# pr - Photorealistic Rendering of Rain Streaks (Garg & Nayar 2006)
'''

cache = {}


class DropType(Enum):
    Big = 0
    Medium = 1
    Small = 2


class Streak:
    def __init__(self, ):
        self.pid = None
        self.world_position_start = None
        self.world_position_end = None
        self.world_diameter_start = None
        self.world_diameter_end = None
        self.image_position_start = None
        self.image_position_end = None
        self.image_diameter_start = None
        self.image_diameter_end = None
        self.ratio = None
        self.max_width = None
        self.length = None
        self.drop_type = None

    def __repr__(self):
        return str(self.__dict__).replace(',', '\n')


class Frame:
    def __init__(self, ):
        self.id = None
        self.starting_time = None
        self.exposure_time = None
        self.streaks_count = None
        self.streaks = None

    def __repr__(self):
        return str(self.__dict__).replace(',', '\n')


class DBManager:
    def __init__(self, streaks_path=None, streaks_path_xml=None, norm_coeff_path=None):
        '''
        Function to initialize the class.
        :param streaks_path: Path to the light texture database.
        :param streaks_path_xml: Path to the output of the simulator.
        :param norm_coeff_path: TODO:: FILL
        '''
        self.streaks_path = streaks_path
        self.streaks_path_xml = streaks_path_xml
        self.streaks_light = np.array([])
        self.norm_coeff_path = norm_coeff_path
        self.streaks_simulator = {}
        self.ratio = np.array([])

    def __repr__(self):
        return "DataAcquisition()"

    def __str__(self):
        return 'DataAcquisition'.format()

    @staticmethod
    def classify_drop(w):
        if w >= 4:
            return DropType(0)
        if w > 1:
            return DropType(1)

        return DropType(2)

    def load_streak_database(self):
        '''
        Function to load and store the texture maps.
        Streaks are stored in a list.
        '''

        if not os.path.exists(self.streaks_path):
            print("No existing path for streak database (", self.streaks_path, ")")
            exit(-1)

        tmp = []
        norm_coeff_path = self.norm_coeff_path
        norm_coeffs = {}

        with open(norm_coeff_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if line[:2] == 'cv':
                coeff = int(line[2:])
                continue
            norm_coeffs.update({coeff: [float(v) for v in line.split('\n')[0].split(' ')[:-1]]})

        for file_name in my_utils.os_listdir(self.streaks_path):
            name = os.path.splitext(file_name)[0]
            coeff, osc = name.split('_')
            if len(coeff) == 3:
                coeff = int(coeff[-1:])
            else:
                coeff = int(coeff[-2:])
            osc = int(osc[-1:])
            drop_image = cv2.imread(os.path.join(self.streaks_path, file_name), cv2.IMREAD_ANYDEPTH)
            drop_image = cv2.cvtColor(drop_image, cv2.COLOR_GRAY2BGR)
            drop_image_norm = ((255.0 * norm_coeffs[coeff][osc] * drop_image) / 65535.0).astype(np.uint8)
            tmp.append(drop_image_norm)
            self.ratio = np.append(self.ratio, tmp[-1].shape[1] / tmp[-1].shape[0])

        self.ratio = np.unique(self.ratio)
        self.streaks_light = np.array(tmp)

    def load_streaks_from_xml(self, dataset, settings, image_shape_WH, use_pickle=True, verbose=True):
        '''
        Function to load and store the output from the physical simulator.
        Streak data is stored in a dictionary. Key: ID, Value: data
        '''
        print('Reading particles file {}'.format(self.streaks_path_xml))

        pickle_version = '1.0'

        # Compute the simulation file hash
        hasher = hashlib.md5()
        with open(self.streaks_path_xml, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        sim_hash = hasher.hexdigest()

        pickle_path = self.streaks_path_xml + '.pkl'
        if use_pickle and os.path.exists(pickle_path):
            print('     loading from pickle')
            input = open(pickle_path, 'rb')
            pickle_data = pickle.load(input)

            # If sim_hash did not change, and image shape is identical too
            if 'version' in pickle_data and pickle_data['version'] == pickle_version and pickle_data[
                'sim_hash'] == sim_hash and np.all(pickle_data['image_shapeWH'] == image_shape_WH):
                self.streaks_simulator = pickle_data['streaks']
                input.close()
                return
            else:
                print('Pickle out-dated. Regenerate.')
                input.close()

        if not os.path.exists(self.streaks_path_xml):
            my_utils.print_error("No existing path for XML file (" + self.streaks_path_xml + ")")
            exit(-1)

        try:
            simulation = parse(self.streaks_path_xml).getroot()
        except Exception as e:
            raise Exception("Reading XML file {} crashed, which is likely due to corrupted particles simulation files. If so, delete this simulation folder manually and re-run to allow generation of new simulation.".format(self.streaks_path_xml))

        if verbose:
            my_utils.print_progress_bar(0, len(simulation))
        try:
            for fix, frame in enumerate(simulation):
                f = Frame()
                f.id = int(frame.attrib['id'])
                f.exposure_time = int(frame.attrib['t'])
                f.starting_time = int(frame.attrib['d'])
                f.streaks_count = int(frame.attrib['rs'])
                f.streaks = {}

                for drop in frame:
                    s = Streak()
                    s.pid = int(drop.attrib["pid"])
                    s.world_position_start = np.array(drop.attrib["wp1"][1:-1].split(';'), dtype=float)
                    s.world_position_end = np.array(drop.attrib["wp2"][1:-1].split(';'), dtype=float)
                    s.world_diameter_start = float(drop.attrib['wd1'])
                    s.world_diameter_end = float(drop.attrib['wd2'])

                    s.image_position_start = np.array(drop.attrib["ip1"][1:-1].split(';'), dtype=float) / settings["render_scale"]  # x,y
                    s.image_position_end = np.array(drop.attrib["ip2"][1:-1].split(';'), dtype=float) / settings["render_scale"]  # x,y
                    s.image_diameter_start = float(drop.attrib['iw1']) / settings["render_scale"]
                    s.image_diameter_end = float(drop.attrib['iw2']) / settings["render_scale"]

                    if dataset == 'nuscenes_gan':
                        # in case the simulation and the rendering are not at the same resolution
                        r = np.mean((image_shape_WH[0] / 1600, image_shape_WH[1] / 900))
                        s.image_position_start = np.array(drop.attrib["ip1"][1:-1].split(';'), dtype=float) * r  # x,y
                        s.image_position_end = np.array(drop.attrib["ip2"][1:-1].split(';'), dtype=float) * r  # x,y
                        s.image_diameter_start = float(drop.attrib['iw1']) * r
                        s.image_diameter_end = float(drop.attrib['iw2']) * r

                    s.image_position_start[1] = image_shape_WH[1] - s.image_position_start[1]
                    s.image_position_end[1] = image_shape_WH[1] - s.image_position_end[1]
                    s.world_position_start[2] *= -1
                    s.world_position_end[2] *= -1
                    diff = abs(s.image_position_start - s.image_position_end)
                    s.max_width = int(max(s.image_diameter_start, s.image_diameter_end))

                    dir1 = np.array([0, -1])
                    dir2 = diff / np.linalg.norm(diff)
                    dir2[1] = -dir2[1]
                    cos_theta = np.dot(dir1, dir2)
                    actual_length = diff[1] / cos_theta
                    s.ratio = s.max_width / actual_length
                    s.image_position_end = s.image_position_end.round().astype(int)
                    s.image_position_start = s.image_position_start.round().astype(int)
                    s.length = np.ceil(np.linalg.norm(s.image_position_start - s.image_position_end)).astype(int)
                    s.drop_type = self.classify_drop(s.max_width)
                    if s.max_width >= 1 and s.length >= 1:
                        f.streaks.update({s.pid: s})

                self.streaks_simulator.update({f.id: f})
                if verbose:
                    my_utils.print_progress_bar(fix + 1, len(simulation))
        except Exception as e:
            ex_type, ex, tb = sys.exc_info()
            my_utils.print_error('Error while parsing XML file.\n\tFile: ' + self.streaks_path_xml)
            traceback.print_tb(tb)
            exit(-1)

    def take_drop_texture(self, drop):
        if drop.ratio < self.ratio[0]:
            drop = self.streaks_light[np.random.randint(0, 10)] / 255.0
            return drop
        if drop.ratio < self.ratio[1]:
            drop = self.streaks_light[np.random.randint(10, 20)] / 255.0
            return drop
        if drop.ratio < self.ratio[2]:
            drop = self.streaks_light[np.random.randint(20, 30)] / 255.0
            return drop
        if drop.ratio < self.ratio[3]:
            drop = self.streaks_light[np.random.randint(30, 40)] / 255.0
            return drop
        else:
            drop = self.streaks_light[np.random.randint(40, 50)] / 255.0
            return drop

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)


class RainRenderer:
    def __init__(self, focal, f_number, focus_plane, radius, fov):
        self.f = focal
        self.N = f_number
        self.focus_plane = focus_plane
        self.radius = radius
        self.fov = fov

    def __repr__(self):
        return "RainRenderer()"

    def __str__(self):
        return 'RainRenderer'.format()

    def circle_of_confusion(self, drop, drop_distance, drop_dict):
        # TODO:: Assess this chunk of code

        # Out-of-focus blur produces larger drop. Thus, we first copy drop in a bigger array and later apply gaussian blur.
        c = abs(self.compute_circle(abs(drop_distance)))
        sigma1, sigma2 = c, c / 2
        # sigma1,sigma2 = (drop_dict.maxWidth/2,drop_dict.maxWidth/2) if drop_dict.dropType==DropType.Big else (drop_dict.maxWidth/3,drop_dict.maxWidth/3)
        shift = int(10 * c)  # TODO: We consider a 10 margin increase, reasonable for most ouf-of-focus blur

        drop2 = cv2.copyMakeBorder(drop, shift, shift, shift, shift, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        drop2 = gaussian_filter(drop2, [sigma1, sigma2, 0])

        return drop2, shift

    @staticmethod
    def warping_points(drop, drop_texture, image_width, image_height):
        x0 = round(drop.image_position_start[0])
        x1 = round(drop.image_position_end[0])
        y0 = round(drop.image_position_start[1])
        y1 = round(drop.image_position_end[1])
        d0 = np.floor(drop.image_diameter_start)
        d1 = np.floor(drop.image_diameter_end)

        minx = max(min(x0, x1), 0)
        miny = max(min(y0, y1), 0)
        maxx = min(max(x0 + d0, x1 + d1), image_width)
        maxy = min(max(y0, y1), image_height)

        # to prevent singularity of pers matrix
        epsilon = 0.001

        p1 = np.float32([
            [0, 0],
            [drop_texture.shape[1], 0],
            [drop_texture.shape[1], drop_texture.shape[0]],
            [0, drop_texture.shape[0]]])

        p2 = np.float32([
            [x0 - minx, y0 - miny],
            [x0 - minx + d0, y0 - miny],
            [x1 - minx + d1 + epsilon, y1 - miny],
            [x1 - minx + epsilon, y1 - miny]])

        return p1, p2, np.array([maxx, maxy]), np.array([minx, miny])

    @staticmethod
    def colour_drop(drop):
        drop[..., :3] = drop[..., :3] * np.expand_dims(drop[..., 3] / 255, axis=-1)
        return drop

    def add_drop_to_image(self, dataset, env_map_xyY, solid_angle_map, drop_fov_pts, drop_minC, bg, rainy_bg,
                          rainy_mask, rainy_saturation_mask, drop, drop_dict, irrad_type, rendering_strategy,
                          opacity_attenuation=1.0):
        global cache
        # This part can be optimized using matplotlib
        # https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
        # TODO - cases where smaller drops may be darker than completely occluding large drops

        exposure_time = db.settings(dataset)["cam_exposure"] / 1000.
        drop_size = 1.16 * 1e-3  # Photorealistic Rendering of Rain Streaks (section 4)

        # Compute the intersection between the FOV points of the drop and the environment map
        # Will return the mask of the drop in the env map
        if rendering_strategy in ['white']:
            # Compute the rain blending
            tau_zero = np.sqrt(drop_size) / 50  # correct vr appendix (10.2) sec
            length_opacity = 1.
            tau_one = exposure_time * length_opacity  # correct #logically
        elif rendering_strategy in ['naive_db']:
            d_avg = (drop_dict.imageDiameterStart + drop_dict.imageDiameterStart) / 2.

            # Compute the rain blending
            tau_zero = np.sqrt(drop_size) / 50  # correct vr appendix (10.2) sec
            length_opacity = d_avg / (drop_dict.length + d_avg)  # doubtful #pr Pg 5 camera effects
            tau_one = exposure_time * length_opacity  # correct #logically
        else:
            # Note: This could perhaps be optimize
            clip = tuple(drop_fov_pts)
            rows, cols = env_map_xyY.shape[:2]
            subj = ((0, 0), (cols, 0), (cols, rows), (0, rows))

            pc = pyclipper.Pyclipper()
            pc.AddPath(clip, pyclipper.PT_CLIP, True)
            pc.AddPath(subj, pyclipper.PT_SUBJECT, True)
            solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

            s = np.asarray(solution[0]).reshape((-1, 2))
            s = np.vstack([s, s[0]])

            # Compute the drop average size
            d_avg = (drop_dict.image_diameter_start + drop_dict.image_diameter_end) / 2.

            # added
            drop_xyY = my_utils.convert_rgb_to_xyY(drop[..., :3])
            drop_xyY[np.isnan(drop_xyY)] = 0

            # Using cache 2.5x faster (Raoul)
            if 'mask_env_float64' not in cache or not np.all(solid_angle_map.shape[:2] == cache['mask_env_bool'].shape[:2]):
                cache['mask_env_float64'] = np.zeros(env_map_xyY.shape[:2], dtype=np.float64)
                cache['mask_env_bool'] = np.zeros(env_map_xyY.shape[:2], dtype=np.bool)

            cache['mask_env_float64'][:] = 0.
            cv2.fillConvexPoly(cache['mask_env_float64'], s, 1)
            cache['mask_env_bool'][:] = cache['mask_env_float64']
            mask_env = cache['mask_env_bool']

            # Get the envmap in drop FOV
            fov_solid_angle = solid_angle_map[mask_env].copy()
            fov_envmap = env_map_xyY[mask_env].copy()
            fov_xyY = (fov_envmap * np.expand_dims(fov_solid_angle, axis=-1)).sum(axis=0)

            fov_xy_avg = fov_xyY[:2] / (np.sum(fov_solid_angle))

            drop_xyY_fov_color = drop_xyY.copy()
            drop_xyY_fov_color[..., :2] = fov_xy_avg

            # In case of drop radiance from environment
            ambient_lum = env_map_xyY[..., 2] * solid_angle_map
            ambient_lum = np.sum(ambient_lum) / np.sum(solid_angle_map)
            if irrad_type == 'ambient':
                # TODO:: check if it was the only irrad_type here
                avg_fov_lum = fov_xyY[..., 2] / np.sum(solid_angle_map)
                drop_Y = 0.94 * avg_fov_lum + 0.06 * ambient_lum
                drop_xyY_fov_color[..., 2] *= drop_Y

            drop_color_rgb = my_utils.convert_xyY_to_rgb(drop_xyY_fov_color)
            drop_color_bgr = drop_color_rgb[..., ::-1]
            drop[..., :3][drop[..., 3] > 0] = drop_color_bgr[drop[..., 3] > 0]

            # Apply defocus effects
            drop, shift = self.circle_of_confusion(drop, drop_dict.world_position_start[2], drop_dict)

            drop_minC_tmp = drop_minC - shift
            drop_minC = np.array([np.clip(drop_minC_tmp[0], 0, bg.shape[1]), np.clip(drop_minC_tmp[1], 0, bg.shape[0])])
            delta = drop_minC - drop_minC_tmp  # evaluate clipping
            drop = drop[:delta[1]] if delta[1] < 0 else drop[delta[1]:]
            drop = drop[:, :delta[0]] if delta[0] < 0 else drop[:, delta[0]:]

            # Compute the rain blending
            tau_zero = np.sqrt(drop_size) / 50  # correct vr appendix (10.2) sec
            length_opacity = opacity_attenuation * d_avg / (drop_dict.length + d_avg)  # pr pg 5 camera effects
            tau_one = exposure_time * length_opacity

        rainy_bg_occ = rainy_bg[drop_minC[1]:drop_minC[1] + drop.shape[0],
                                drop_minC[0]:drop_minC[0] + drop.shape[1], :].copy()
        rainy_mask_occ = rainy_mask[drop_minC[1]:drop_minC[1] + drop.shape[0],
                                    drop_minC[0]:drop_minC[0] + drop.shape[1]].copy()
        rainy_sat_mask_occ = rainy_saturation_mask[drop_minC[1]:drop_minC[1] + drop.shape[0],
                                                   drop_minC[0]:drop_minC[0] + drop.shape[1]].copy()

        # Blending is directly applied on the rainy_bg. Hence, no conditions for application is later required.
        # Which seems more correct and faster.
        # The bias is that it is drop-order dependent
        drop_vis = drop[:rainy_bg_occ.shape[0], :rainy_bg_occ.shape[1]]
        drop_vis_alpha = drop_vis[:, :, 3]
        drop_vis_alpha_ = np.expand_dims(drop_vis_alpha, axis=-1)

        rainy_bg_occ = ((1. - ((drop_vis_alpha_ * tau_one) / exposure_time)) * rainy_bg_occ) + drop_vis[:, :, :3] * (
                    tau_one / tau_zero)

        rainy_bg_occ = np.clip(rainy_bg_occ, 0, 1)
        drop_blend = rainy_bg_occ
        # -------------

        rainy_mask_occ += drop_vis_alpha

        drop_viz_sat = drop_vis[..., :3].copy()
        rainy_sat_mask_occ += np.clip(drop_viz_sat.astype(np.float64), 0, 1)

        rainy_bg[drop_minC[1]:drop_minC[1] + drop_blend.shape[0],
                 drop_minC[0]:drop_minC[0] + drop_blend.shape[1]] = rainy_bg_occ
        rainy_mask[drop_minC[1]:drop_minC[1] + drop.shape[0],
                   drop_minC[0]:drop_minC[0] + drop.shape[1]] = rainy_mask_occ
        rainy_saturation_mask[drop_minC[1]:drop_minC[1] + drop.shape[0],
                              drop_minC[0]:drop_minC[0] + drop.shape[1]] = rainy_sat_mask_occ

        return rainy_bg, rainy_mask, rainy_saturation_mask, drop_vis, drop_blend, drop_minC

    def compute_circle(self, o, is_infinity=False):
        if is_infinity:
            return self.f ** 2 / (self.N * o)
        else:
            result = ((o - self.focus_plane) * self.f ** 2) / (o * (self.focus_plane - self.f) * self.N)
            return result / 4.65e-06

    @staticmethod
    def imshow_with_alpha(win_name, im):
        '''
        Function to show an image with alpha channel.
        :param win_name: Name of the window.
        :param im: Image to show.
        '''

        cv2.imshow(win_name, (im[..., 0:3] * (cv2.cvtColor(im[..., 3], cv2.COLOR_GRAY2BGR) / 255.)).astype(np.uint8))
        cv2.waitKey(0)

    @staticmethod
    def make_rain_layer(drop, blended_drop, rain_layer, mask, drop_min_C):
        layer_considered = rain_layer[drop_min_C[1]:drop_min_C[1] + drop.shape[0],
                                      drop_min_C[0]:drop_min_C[0] + drop.shape[1]].copy()
        mask_considered = mask[drop_min_C[1]:drop_min_C[1] + drop.shape[0],
                               drop_min_C[0]:drop_min_C[0] + drop.shape[1]].copy()

        layer_considered[:, :, 3][mask_considered[:, :] > 0] = 255
        layer_considered[:, :, :3][mask_considered[:, :] > 0] = np.maximum(
            layer_considered[:, :, :3][mask_considered[:, :] > 0], blended_drop[mask_considered[:, :] > 0])

        rain_layer[drop_min_C[1]:drop_min_C[1] + drop.shape[0],
        drop_min_C[0]:drop_min_C[0] + drop.shape[1]] = layer_considered
        return rain_layer

    @staticmethod
    def merge_layers(bg, over):
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
        open_cv_image = bg.copy()
        threshold = 40

        for i in range(over.shape[0]):
            for j in range(over.shape[1]):
                if over[i, j, 3] > threshold:
                    open_cv_image[i, j] = over[i, j]
                if over[i, j, 3] > 0:
                    alpha = over[i, j, 3] / 255.0
                    open_cv_image[i, j] = alpha * over[i, j] + ((1 - alpha) * bg[i, j])

        return open_cv_image

    @staticmethod
    def show(rain, background, split=True):
        if split:
            cv2.imshow('Rain Layer', rain)
            cv2.imshow('Result', background)
            cv2.waitKey(0)
        else:
            cv2.imshow('Result', np.vstack((background[..., 0:3], rain[..., 0:3])))
            cv2.waitKey(0)


class FovComputation:
    def __init__(self, camera):
        self.camera = camera

    @staticmethod
    def normalize(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def rotation_matrix(axis, theta):
        axis = np.asarray(axis)
        c, s = np.cos(theta), np.sin(theta)
        skv = np.roll(np.roll(np.diag(axis.flatten()), 1, 1), -1, 0)
        m3 = (c * np.identity(3)) + s * (skv - skv.T) + ((1 - c) * np.outer(axis, axis))
        return m3

    @staticmethod
    def intersection_sphere(position, direction, radius):
        dx = direction[0]
        dy = direction[1]
        dz = direction[2]

        x0 = position[0]
        y0 = position[1]
        z0 = position[2]
        R = radius
        cx = cy = cz = 0

        a = dx * dx + dy * dy + dz * dz
        b = 2 * dx * (x0 - cx) + 2 * dy * (y0 - cy) + 2 * dz * (z0 - cz)
        c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0 + \
            -2 * (cx * x0 + cy * y0 + cz * z0) - R * R

        disc = b ** 2 - 4 * a * c
        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2 * a)
        # t2 = (-b - sqrt_disc) / (2 * a)  # No need to compute the other solution

        # x = position[0] + t1 * dx
        # y = position[1] + t1 * dy
        # z = position[2] + t1 * dz

        r1 = position + (t1 * direction)
        # r2 = position + (t2 * direction)
        return r1  # , r2

    @staticmethod
    def cart2sph(inpoint):
        x = inpoint[0]
        y = inpoint[1]
        z = inpoint[2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        el = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))  # Was an error here
        az = np.arctan2(y, x)
        if az < 0:
            az += 2 * np.pi
        if el < 0:
            el += 2 * np.pi
        if az > np.pi * 2:
            az -= 2 * np.pi
        if el > np.pi * 2:
            el -= 2 * np.pi
        return az, el, r

    @staticmethod
    def make_surface(theta0, theta1, phi0, phi1, radius, resolution=25):
        THETA, PHI = np.meshgrid(np.linspace(theta0, theta1, resolution), np.linspace(phi0, phi1, resolution))
        X = radius * np.sin(PHI) * np.cos(THETA)
        Y = radius * np.sin(PHI) * np.sin(THETA)
        Z = radius * np.cos(PHI)
        return X, Y, Z

    def compute_fov_plane_points(self, drop_dict, radius, fov, N, env_shape):
        try:
            drop_position = np.array((drop_dict.world_position_start + drop_dict.world_position_end) / 2)
            drop_position[1], drop_position[2] = drop_position[2], drop_position[1].copy()
            drop_direction = self.normalize(drop_position - self.camera)

            theta = np.deg2rad(fov / 2)

            # 2 Compute the plane for which u is the normal and lies in
            a = drop_direction[0]
            b = drop_direction[1]
            c = drop_direction[2]
            d = np.dot(drop_position, drop_direction)
            if b == 0:
                b = 0.001

            # 3 Find point P on the plane
            px = drop_position[1]
            pz = 0
            py = (-a * px + d - c * pz) / b

            point = np.array([px, py, pz])

            # Compute U = p-drop_dict
            u = self.normalize(drop_position - point)
            assert (np.all(~np.isnan(u)) and "Some values are NAN")

            # 4 Compute v so angle between v-n is FOV/2
            rot_vec = np.cross(u, drop_direction)
            rot_mat = self.rotation_matrix(rot_vec, -theta)

            v = np.dot(drop_direction, rot_mat)

            # 5 Rotate v along dropdirection
            phi = np.arange(0, 2 * np.pi, (2 * np.pi) / N)
            vectors = np.array([])
            for angle in phi:
                M = self.rotation_matrix(drop_direction, angle)
                vectors = np.append(vectors, [np.dot(v, M)])

            vectors = np.reshape(vectors, (-1, 3))

            # 6 Intersections
            points = np.array([])
            for dir_v in vectors:
                points = np.append(points, self.intersection_sphere(drop_position, dir_v, radius))
            points = np.reshape(points, (-1, 3))

            # 3D TO PLANE MAPPING
            azs = np.array([])
            points_image = np.array([])
            for p in points:
                azimuth, elevation, r = self.cart2sph(p)

                # Convert to the image encoding azimuth angle and shift: [-pi/2, 3*pi/2]
                azimuth = ((2 * np.pi - azimuth) - np.pi / 2)
                # Modulo to convert to azimuth: [0, 2*pi]
                azimuth = azimuth % (2 * np.pi)
                # Convert to homogeneous UV coord [0, 1]
                u = azimuth / (2 * np.pi)

                # Coordinate space change
                elevation = (elevation + np.pi / 2)
                # Modulo to convert to elevation: [0, 2*pi]
                elevation = elevation % (2 * np.pi)
                v = 1. - elevation / np.pi

                azs = np.append(azs, azimuth)
                points_image = np.append(points_image, [u * env_shape[1], v * env_shape[0]])

            points_image = np.reshape(points_image, (-1, 2))
            azs = np.append(azs, azs[0])

            cond = np.bitwise_or(np.isclose(np.diff(azs), 0), np.diff(azs) < 0)
            cond_true = cond
            cond_false = ~cond
            count_true = np.sum(cond_true)
            count_false = np.sum(cond_false)

            pos_true = np.where(cond_true)[0][0]
            pos_false = np.where(cond_false)[0][0]
            rows, cols = env_shape[:2]
            if count_true == 1:  # top
                final_pts = np.vstack([points_image[:pos_true + 1],
                                      [cols, points_image[pos_true][1]],
                                      [cols, 0],
                                      [0, 0],
                                      [0, points_image[np.mod(pos_true + 1, N)][1]],
                                       points_image[pos_true + 1:]])  # Top left

            elif count_false == 1:  # bottom
                final_pts = np.vstack([points_image[:pos_false + 1],
                                      [0, points_image[pos_false][1]],
                                      [0, rows],
                                      [cols, rows],
                                      [cols, points_image[np.mod(pos_false + 1, N)][1]],
                                       points_image[pos_false + 1:]])  # bot left

            else:
                final_pts = points_image

            return np.array(final_pts), points, drop_position, drop_direction
        except:
            drop_position = np.array((drop_dict.world_position_start + drop_dict.world_position_end) / 2)
            drop_position[1], drop_position[2] = drop_position[2], drop_position[1].copy()
            drop_direction = self.normalize(drop_position - self.camera)
            print('Drop skipped')

            return np.array([]), np.array([]), drop_direction, drop_position


class EnvironmentMapGenerator:
    def __init__(self, f, image_width, image_height):
        # http://answers.opencv.org/question/17076/conversion-focal-distance-from-mm-to-pixels/
        self.image_width = image_width
        self.image_height = image_height
        self.focal = int(((f * 1000) / 12.7) * image_width)
        self.fillMatUp = 0
        self.fillMatDown = 0

    def convert2cyl(self, xyz, center):
        s = self.focal
        x = s * np.arctan(xyz[0] / self.focal) + center[0]
        y = s * (xyz[1] / np.sqrt(xyz[0] ** 2 + self.focal ** 2)) + center[1]
        return np.array([round(x), round(y)])

    def convert2cyl_whole(self, xyz_all, center):
        mod_xy = np.zeros((xyz_all.shape[0], xyz_all.shape[1], 2))
        mod_xy[:, :, 0] = (self.focal * (xyz_all[:, :, 0] /
                                         (np.sqrt(xyz_all[:, :, 1] ** 2 + self.focal ** 2)))) + center[1]
        mod_xy[:, :, 1] = (self.focal * np.arctan(xyz_all[:, :, 1] / self.focal)) + center[0]

        return mod_xy

    def max_coord(self, center):
        s = self.focal
        x = s * np.arctan(center[0] / self.focal) + center[0]
        y = s * (center[1] / np.sqrt(center[0] ** 2 + self.focal ** 2)) + center[1]
        return round(x), round(y)

    def min_coord(self, center):
        s = self.focal
        x = s * np.arctan(-center[0] / self.focal) + center[0]
        y = s * (-center[1] / np.sqrt(center[0] ** 2 + self.focal ** 2)) + center[1]
        return round(x), round(y)

    def generate_map(self, background):
        # Easier if everything is in int due to cv2 calls
        background = (background * 255).astype(np.uint8)
        center = np.array([int(background.shape[1] // 2), int(background.shape[0] // 2)])

        max_x, max_y = self.max_coord(center)
        min_x, min_y = self.min_coord(center)
        mask = np.zeros((background.shape[0], int(max_x - min_x) + 1), np.uint8)

        # creating co-ord matrix
        xyz_all = np.zeros((background.shape[0], background.shape[1], 3))
        x = np.linspace(0, background.shape[1] - 1, background.shape[1])
        y = np.linspace(0, background.shape[0] - 1, background.shape[0])
        xx, yy = np.meshgrid(x, y)
        xyz_all[:, :, 0] = yy - center[1]
        xyz_all[:, :, 1] = xx - center[0]

        # fish-eye co-ord mapping
        xy = np.round(self.convert2cyl_whole(xyz_all, center))
        xy[:, :, 1] = xy[:, :, 1] - min_x
        ind_vals, ind = np.unique(xy.astype(np.int32).reshape(-1, 2), axis=0, return_index=True)

        # creating fish-eye image
        if background.ndim == 3:
            cyl = np.zeros((background.shape[0], int(max_x - min_x) + 1, 3), np.uint8)
            cyl[ind_vals[:, 0], ind_vals[:, 1]] = background.reshape((-1, 3))[ind]
        else:
            cyl = np.zeros((background.shape[0], int(max_x - min_x) + 1), np.float64)
            cyl[ind_vals[:, 0], ind_vals[:, 1]] = background.reshape((-1))[ind]

        mask[ind_vals[:, 0], ind_vals[:, 1]] = 255

        self.fillMatUp, self.fillMatDown = self.fill_matrices(cyl, mask)

        # filling the bottom portion
        mask_temp = cv2.flip(mask, 0)[:mask.shape[0] // 2, :]
        cyl_temp = cv2.flip(cyl, 0)[:cyl.shape[0] // 2, :]
        cyl_temp[np.where(mask_temp == 0)[0], np.where(mask_temp == 0)[1]] = cv2.flip(cyl, 0)[
            self.fillMatDown[:, 0], self.fillMatDown[:, 1]]
        cyl[-cyl_temp.shape[0]:, :] = cv2.flip(cyl_temp, 0)

        # filling gaps
        # filling the top portion
        mask_temp = mask[:mask.shape[0] // 2, :]
        cyl_temp = cyl[:cyl.shape[0] // 2, :]
        cyl_temp[np.where(mask_temp == 0)[0], np.where(mask_temp == 0)[1]] = cyl[
            self.fillMatUp[:, 0], self.fillMatUp[:, 1]]
        cyl[:cyl_temp.shape[0], :] = cyl_temp

        result = cv2.copyMakeBorder(cyl, 0, 0, int(cyl.shape[1] / 2), int(cyl.shape[1] / 2), cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))  # TODO: check issue with float values ?
        mask_result = cv2.copyMakeBorder(mask, 0, 0, int(mask.shape[1] / 2), int(mask.shape[1] / 2),
                                         cv2.BORDER_CONSTANT, value=0)  # TODO: check issue with float values ?

        # attaching cropped flipped image to left image and mask
        side = cyl[:, 0:int(cyl.shape[1] / 2)]
        side = cv2.flip(side, 1)  # TODO: check issue with float values ?
        result[:, 0:side.shape[1]] = side

        mask_side = mask[:, :cyl.shape[1] // 2]
        mask_side = cv2.flip(mask_side, 1)  # TODO: check issue with float values ?
        mask_result[:, :mask_side.shape[1]] = mask_side

        # attaching cropped flipped image to right
        side = cyl[:, cyl.shape[1] // 2:]
        side = cv2.flip(side, 1)  # TODO: check issue with float values ?
        result[:, result.shape[1] - side.shape[1]:result.shape[1]] = side

        mask_side = mask[:, cyl.shape[1] // 2:]
        mask_side = cv2.flip(mask_side, 1)  # TODO: check issue with float values ?
        mask_result[:, mask_result.shape[1] - side.shape[1]:] = mask_side

        if result.ndim == 3:
            blur = cv2.GaussianBlur(result, (15, 15), 0)  # TODO: check issue with float values ?
            mask_result = np.tile(np.expand_dims(mask_result, axis=-1), (1, 1, 3))
            result = result + ((blur - result) & ~mask_result)

        return result / 255.0

    @staticmethod
    def fill_matrices(cyl, mask):
        # co-ord for filling upper part

        mask_temp = mask[:mask.shape[0] // 2, :]
        cyl_temp = cyl[:cyl.shape[0] // 2, :]
        y_fill = np.argmax(mask_temp > 0, axis=0)
        x_fill = np.arange(cyl_temp.shape[1])
        xy_fill = np.concatenate([np.expand_dims(y_fill, axis=-1), np.expand_dims(x_fill, axis=-1)], axis=-1)

        # indices which are empty
        ind_not_filled = np.where(mask_temp == 0)

        fill_mat_up = np.zeros((ind_not_filled[0].shape[0], 2)).astype(np.int)
        for i in range(fill_mat_up.shape[0]):
            x_ind = ind_not_filled[1][i]
            fill_mat_up[i] = xy_fill[x_ind]

        mask_temp = cv2.flip(mask[mask.shape[0] // 2:, :], 0)
        cyl_temp = cv2.flip(cyl[cyl.shape[0] // 2:, :], 0)
        y_fill = np.argmax(mask_temp > 0, axis=0)
        x_fill = np.arange(cyl_temp.shape[1])
        xy_fill = np.concatenate([np.expand_dims(y_fill, axis=-1), np.expand_dims(x_fill, axis=-1)], axis=-1)

        # indices which are empty
        ind_not_filled = np.where(mask_temp == 0)

        fill_mat_down = np.zeros((ind_not_filled[0].shape[0], 2)).astype(np.int)
        for i in range(fill_mat_down.shape[0]):
            x_ind = ind_not_filled[1][i]
            fill_mat_down[i] = xy_fill[x_ind]

        return fill_mat_up, fill_mat_down

    def generate_depth_env_maps(self, depth_maps):

        depth_env_maps = []

        h_depth_map, w_depth_map = depth_maps[0].shape
        center = np.array([int(depth_maps.shape[2] / 2), int(depth_maps.shape[1] / 2)])

        # getting max and min co-ords of the fish eye image
        max_x, max_y = self.max_coord(center)
        min_x, min_y = self.min_coord(center)

        cyl = np.zeros((h_depth_map, int(max_x - min_x) + 1), np.float32)
        mask = np.zeros((cyl.shape[0], cyl.shape[1]))
        xyz_all = np.zeros((h_depth_map, w_depth_map, 3))

        # creating co-ord matrix
        x = np.arange(w_depth_map)
        y = np.arange(h_depth_map)
        xx, yy = np.meshgrid(x, y)
        xyz_all[:, :, 0] = yy - center[1]
        xyz_all[:, :, 1] = xx - center[0]

        # fish-eye co-ord mapping
        xy = np.round(self.convert2cyl_whole(xyz_all, center))
        xy[:, :, 1] = xy[:, :, 1] - min_x
        ind_vals, ind = np.unique(xy.astype(np.int32).reshape(-1, 2), axis=0, return_index=True)

        cyl[ind_vals[:, 0], ind_vals[:, 1]] = depth_maps[0].reshape((-1))[ind]
        mask[ind_vals[:, 0], ind_vals[:, 1]] = 255

        for depthMapNo in range(depth_maps.shape[0]):
            cyl[ind_vals[:, 0], ind_vals[:, 1]] = depth_maps[depthMapNo].reshape((-1))[ind]
            mask[ind_vals[:, 0], ind_vals[:, 1]] = 255

            # fill the top part
            mask_temp = mask[:mask.shape[0] // 2, :]
            cyl_temp = cyl[:cyl.shape[0] // 2, :]
            cyl_temp[np.where(mask_temp == 0)[0], np.where(mask_temp == 0)[1]] = cyl[
                self.fillMatUp[:, 0], self.fillMatUp[:, 1]]
            cyl[:cyl.shape[0] // 2, :] = cyl_temp

            # fill the bottom part
            mask_temp = cv2.flip(mask, 0)[:mask.shape[0] // 2, :]
            cyl_temp = cv2.flip(cyl, 0)[:cyl.shape[0] // 2, :]
            cyl_temp[np.where(mask_temp == 0)[0], np.where(mask_temp == 0)[1]] = cv2.flip(cyl, 0)[
                self.fillMatDown[:, 0], self.fillMatDown[:, 1]]
            cyl[cyl.shape[0] // 2:, :] = cv2.flip(cyl_temp, 0)

            # expanding the map by pasting flipped cropped images on left and right
            result = cv2.copyMakeBorder(cyl, 0, 0, int(cyl.shape[1] / 2), int(cyl.shape[1] / 2), cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))

            # paste left
            side = cyl[:, 0:int(cyl.shape[1] / 2)]
            side = cv2.flip(side, 1)
            result[:, 0:side.shape[1]] = side

            # paste right
            side = cyl[:, int(cyl.shape[1] / 2):cyl.shape[1]]
            side = cv2.flip(side, 1)
            result[:, result.shape[1] - side.shape[1]:result.shape[1]] = side
            depth_env_maps.append(result)

        return np.array(depth_env_maps)

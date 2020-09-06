import math
import sys

import cv2
import numpy as np

sys.path.append("..")


#######################################################################################
# focal number changed from 6 to 8.5 as focal number varies from 1.5 - 16 in the manual
# angle changed to 80

## formula followed for generation of the fog like rain
'''
beta_ext = 0.312 * rain_intensity (R) ^ 0.67
f_ext = e ^ (- beta_ext * depth)

g = [0.9,1.0]
beta_hg = (1 - g)^2/4*pi*((1 - g^2 - 2g*cos \theta)^1.5)
l_in = beta_hg * e_sun * (1 - f_ext)

l = l_0*f_ext + l_in

'''
class FogRain:
    def __init__(self, rain_intensity, focal, f_number, angle, exposure=2, camera_gain=20):
        self.rain_intensity = rain_intensity
        self.angle = angle
        self.focal = focal
        self.f_number = f_number
        self.beta_ext = 0
        self.exposure_time = exposure * 1e-3
        # gain is the camera gain for camera (Flea2 used in kitti, acA1600-60gm/gc in NuScenes)
        self.camera_gain = camera_gain

        self.current_image = np.array([])
        self.current_depth = np.array([])

    def calc_beta_ext(self):
        # A multiscale model for rain rendering in real-time (Weber 2015)
        rain_intensity = self.rain_intensity
        return 0.312 * rain_intensity ** 0.67

    def calc_f_ext(self):
        self.beta_ext = self.calc_beta_ext()
        # depth in km as beta_ext is in km^-1
        f_ext = np.exp((-self.beta_ext) * (self.current_depth / 1000))
        return np.tile(np.expand_dims(f_ext, axis=-1), (1, 1, 3))

    def calc_irradiance(self):
        # According to Garg and Nayar (Basic Principles of Imaging and Photometry)
        irradiance = (4 * (self.f_number ** 2) * self.current_image) / (self.exposure_time * self.camera_gain * np.pi)
        return irradiance

    @staticmethod
    def calc_g():
        return 0.97

    def calc_beta_hg(self):
        # A multiscale model for rain rendering in real-time (Weber 2015), Eq. 10
        g = self.calc_g()
        cos_term = math.cos(math.radians(self.angle))
        return (1 - (g ** 2)) / (4 * np.pi * ((1 + g ** 2 - 2 * g * cos_term) ** 1.5))

    def calc_l_in(self):
        beta_hg = self.calc_beta_hg()
        f_ext = self.calc_f_ext()
        irradiance = self.calc_irradiance()
        irradiance_mean = np.mean(irradiance.reshape(-1, 3), axis=0)
        l_in = beta_hg * irradiance_mean * (1 - f_ext)
        l_in = np.clip(l_in, 0, 1)
        return l_in

    def calc_l(self):
        f_ext = self.calc_f_ext()
        l_in = self.calc_l_in()

        f_ext = cv2.GaussianBlur(f_ext, (25, 25), 25)
        l_in = cv2.GaussianBlur(l_in, (25, 25), 25)

        real_image = self.current_image

        # A multiscale model for rain rendering in real-time (Weber 2015), Eq. 13
        l = real_image * f_ext + l_in
        l = np.clip(l, 0, 1)
        return l

    def fog_rain_layer(self, image, depth):
        self.current_image = image.copy()
        self.current_depth = depth.copy()

        simulated_image = np.clip(self.calc_l(), 0, 1)

        return simulated_image


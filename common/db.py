import importlib
import os
import re
import numpy as np

from common import my_utils

_settings_defaults = {}
# Camera intrinsic parameters
_settings_defaults["cam_hz"] = 10  # Camera Hz (aka FPS)
_settings_defaults["cam_CCD_WH"] = [1242, 375]  # Camera CDD Width and Height (pixels)
_settings_defaults["cam_CCD_pixsize"] = 4.65  # Camera CDD pixel size (micro meters)
_settings_defaults["cam_WH"] = [1242, 375]  # Camera image Width and Height (pixels)
_settings_defaults["cam_focal"] = 6  # Focal length (mm)
_settings_defaults["cam_gain"] = 20  # Camera gain
_settings_defaults["cam_f_number"] = 6.0  # F-Number
_settings_defaults["cam_focus_plane"] = 6.0  # Focus plane (meter)
_settings_defaults["cam_exposure"] = 2  # Camera exposure (ms)

# Camera extrinsic parameters (right-handed coordinate system)
_settings_defaults["cam_pos"] = [1.5, 1.5, 0.3]  # Camera pos (meter)
_settings_defaults["cam_lookat"] = [1.5, 1.5, -1.]  # Camera look at vector (meter)
_settings_defaults["cam_up"] = [0., 1., 0.]  # Camera up vector (meter)

# Renderer specific
_settings_defaults["depth_scale"] = 1  # Depth scale w.r.t. image size. depth_scale = depth_size / image_size
_settings_defaults["render_scale"] = 1  # Scale down applied at rendering (1 = full res, N = original_size / N). I.e. 2 means half size

# These are the sizes used during the rendering process:
# image_size = original_image_size // settings["render_scale"]
# depth_size = (original_depth_size * settings["depth_scale"]) // settings["render_scale"]
#
# Example 1: images are 1024x512 and depthmaps are 1024x512, set: depth_scale = 1, render_scale = 1 (default values)
#               => output size will be 1024x512
# Example 2: images are 1024x512 and depthmaps are 512x256, set: depth_scale = 2, render_scale = 1
#               => output size will be 1024x512
# Example 3: images 2048x1024 and depthmaps are 512x256 and you wish to downscale the output, use: depth_scale = 4, render_scale = 4
#               => output size will be 512x256
#
# IMPORTANT note: if depth and image size mismatch, the depth is supposed to be crop-centered
# (which often occur with deep depth estimation, due to padding)

# Particle simulation parameters
# sim_* settings will be passed along to the particles simulator, they allow controlling the physical simulation.
# There are two mode of simulation: normal and steps
#
# normal:   simulate a particle event (rain) for a given duration "sim_duration" (sec). Useful: to generate for example 25mm/hr
# steps:    simulate a particle event, while applying step-wise changes to the simulation.
#           Step duration depends on the camera frequency. Suppose 10Hz, the step then last 100 ms.
#           At each step parameters are adjusted, which allow for example simulating rain of increasing intensity,
#           or rain seen at different motion speed (ideal to user along with real acquisition motion speed
#           (e.g. from odometry/GPS).
#           The
#
#           While there are many customizable parameters, the current particle simulator wrapper supports:
#               cam_motion: list of camera motion speeds (km/h), useful to mimic data recording conditions (e.g. from odometry/gps data)
#               cam_exposure: list of camera exposure times (ms)
#               cam_focal: list of camera focals (mm)
#               rain_fallrate: list of rain fallrates (mm/hr), useful to mimic changing rain intensity (e.g. from light: 5mm/hr to heavy 100mm/hr)
#
#           At simulation step i, the ith parameter of above customizable parameters is applied (if it exists),
#           and remains applied unless later changed.

_settings_defaults["sim_hz"] = 2000  # Update frequency (Hz) of the time-discrete discrete particle simulator. Lowering this number may significantly speed up, but also lower simulation precision. We do not recommand value below 1000. / cam_exposure as particles may not be updated during camera shutter opening.
_settings_defaults["sim_mode"] = "normal"  # normal|steps (refer to above help)
_settings_defaults["sim_duration"] = 34.  # Simulation duration (sec), will be overridden if "steps" is provided
_settings_defaults["sim_steps"] = {}  # Contains dictionary of steps parameters (refer to above help)

# Sequence specific
_settings_defaults["sequences"] = {}


dbs = {}
def _load_db(db):
    return importlib.import_module("config." + db)

def _db(db):
    if not db in dbs:
        dbs[db] = _load_db(db)

    return dbs[db]

def resolve_paths(db, results):
    results = _db(db).resolve_paths(results)

    assert "images" in results
    assert "depth" in results
    assert "calib" in results, "calib files are missing (Kitti format), if no calibration files are provided just set None for each sequence."

    return results

def settings(db):
    db = _db(db)

    settings = {**_settings_defaults, **db.settings()}

    # Correct any forward / or backslash \ to be OS compliant
    settings["sequences"] = {my_utils.path_os_s(s): settings["sequences"][s] for s in settings["sequences"]}

    assert_settings(settings)
    return settings

def assert_settings(settings):
    assert settings["render_scale"] >= 1 and isinstance(settings["render_scale"], int)
    assert settings["cam_exposure"] <= 1000./settings["cam_hz"], "Exposure should be lower than 1000./Hz otherwise camera frames temporally overlaps (non-tested behavior, remove assertion at your own risk)."
    assert settings["cam_lookat"][2] < 0, "Z axis should be negative (other systems were not tested, remove assertion at your own risk)."
    assert np.isclose(np.linalg.norm(settings["cam_up"]), 1), "cam_up must be of norm 1"

def sim(db_s, seq, particles_root):
    db_settings = settings(db_s)
    sim = {"path": os.path.join(particles_root, seq), "options": db_settings.copy()}

    # Find ad-hoc settings
    settings_seq = [s for s in db_settings["sequences"] if re.match(s.replace("\\", "\\\\"), seq) is not None]
    if len(settings_seq) > 0:
        sim["path"] = os.path.join(particles_root, settings_seq[0].replace("*", "x"))
        sim["options"] = {**sim["options"], **db_settings["sequences"][settings_seq[0]]}
        del sim["options"]["sequences"]  # No need to keep
    else:
        print(" No specific simulation settings found for '{}'. Will fallback to database '{}' settings, if not intentional this might fails.".format(seq, db_s))

    return sim
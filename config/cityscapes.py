import os

import glob2
import numpy as np


def _sequences(params):
    # Recursively find all children folders
    sequences = np.array([x[0][len(params.images_root) + 1:] for x in os.walk(os.path.join(params.images_root))])

    # A folder is a sequence if it has some PNGs inside and is not a depth folder
    cond1 = [len(glob2.glob(os.path.join(params.images_root, p, '*.png'))) != 0 for p in sequences]
    cond2 = ['depth' not in p.split(os.sep)[-2:] for p in sequences]  # Verify it's not a depth folder

    return sequences[np.bitwise_and(cond1, cond2)]


def resolve_paths(params):
    params.sequences = _sequences(params)
    assert (len(params.sequences) > 0), "There are no valid sequences folder in the dataset root. Have you altered cityscapes file structure ?"

    # Setting images dir of sequence
    params.images = {sequence: os.path.join(params.images_root, sequence) for sequence in params.sequences}
    params.depth = {sequence: os.path.join(params.depth_root, sequence, os.pardir, 'depth', sequence.split(os.sep)[-1]) for sequence in params.sequences}
    params.calib = {sequence: None for sequence in params.sequences}  # We ignore camera intrinsic calib (non provided), this makes only little differences

    return params

def settings():
    settings = {}
    settings["cam_hz"] = 10   # Camera Hz (aka FPS)
    settings["cam_CCD_WH"] = [2040, 1016]  # Camera CDD Width and Height (pixels)
    settings["cam_CCD_pixsize"] = 2.2  # Camera CDD pixel size (micro meters)
    settings["cam_WH"] = [2040, 1016]  # Camera image Width and Height (pixels)
    settings["cam_focal"] = 6  # Focal length (mm)
    settings["cam_gain"] = 20
    settings["cam_f_number"] = 6.0  # F-Number
    settings["cam_focus_plane"] = 6.0  # Focus plane (meter)
    settings["cam_exposure"] = 5.0  # Camera exposure (ms)

    settings["depth_scale"] = 2  # Ratio: image_size / depth_size. For Cityscapes, we assume depth is half resolution of rgb
    settings["render_scale"] = 2  # Rendering scale (here, halve size to speed up the process, Cityscapes is really big)

    # Camera extrinsic parameters
    settings["cam_pos"] = [1.5, 1.5, 0.3]  # Camera pos
    settings["cam_lookat"] = [1.5, 1.5, -1.]  # Camera look at vector
    settings["cam_up"] = [0., 1., 0.]  # Camera up vector

    # Sequence-wise settings
    # Note: sequence object and settings are merged, hence any setting can be overwriten sequence-wise
    settings["sequences"] = {}

    # For sequences in leftImg8bit
    settings["sequences"]["leftImg8bit"] = {}
    settings["sequences"]["leftImg8bit"]["sim_mode"] = "steps"
    settings["sequences"]["leftImg8bit"]["sim_steps"] = {"cam_motion": np.arange(50., 0.-1, -1)}  # Since data_object sequence lack speed data, we assume random speed between 50km/hr and 0km/hr (european city speed regulation)

    return settings

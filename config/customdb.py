# Sample database configuration file

import os

def resolve_paths(params):
    # List sequences path (relative to dataset folder)
    # Let's just consider any subfolder is a sequence
    params.sequences = [x for x in os.listdir(params.images_root) if os.path.isdir(os.path.join(params.images_root, x))]
    assert (len(params.sequences) > 0), "There are no valid sequences folder in the dataset root"

    # Set source image directory
    params.images = {s: os.path.join(params.dataset_root, s, 'rgb') for s in params.sequences}

    # Set calibration (Kitti format) directory IF ANY (optional)
    params.calib = {s: None for s in params.sequences}

    # Set depth directory
    params.depth = {s: os.path.join(params.dataset_root, s, 'depth') for s in params.sequences}

    return params

def settings():
    settings = {}

    # Camera intrinsic parameters
    settings["cam_hz"] = 10               # Camera Hz (aka FPS)
    settings["cam_CCD_WH"] = [1242, 375]  # Camera CDD Width and Height (pixels)
    settings["cam_CCD_pixsize"] = 4.65    # Camera CDD pixel size (micro meters)
    settings["cam_WH"] = [1242, 375]      # Camera image Width and Height (pixels)
    settings["cam_focal"] = 6             # Focal length (mm)
    settings["cam_gain"] = 20             # Camera gain
    settings["cam_f_number"] = 6.0        # F-Number
    settings["cam_focus_plane"] = 6.0     # Focus plane (meter)
    settings["cam_exposure"] = 2          # Camera exposure (ms)

    # Camera extrinsic parameters (right-handed coordinate system)
    settings["cam_pos"] = [1.5, 1.5, 0.3]     # Camera pos (meter)
    settings["cam_lookat"] = [1.5, 1.5, -1.]  # Camera look at vector (meter)
    settings["cam_up"] = [0., 1., 0.]         # Camera up vector (meter)

    # Sequence-wise settings
    # Note: sequence object and settings are merged, hence any setting can be overwritten sequence-wise
    settings["sequences"] = {}

    # Sequence "seq1" will just run a normal rain simulation for 10sec
    settings["sequences"]["seq1"] = {}
    settings["sequences"]["seq1"]["sim_mode"] = "normal"
    settings["sequences"]["seq1"]["sim_duration"] = 10  # Duration of the rain simulation (sec)

    # Sequence "seq2" will simulate rain seen at varying motion speed from 100km/hr to 0km/hr. 1st frame will be 100km/hr, 2nd 90km/hr, ..., Nth 0km/hr. In general, the ith frame uses i % N simulation (% = modulo)
    settings["sequences"]["seq2"] = {}
    settings["sequences"]["seq2"]["sim_mode"] = "steps"
    settings["sequences"]["seq2"]["sim_steps"] = {"cam_motion": [100., 90., 80., 70., 60., 50., 40., 30., 20., 10., 0.]}  # Camera motion in km/hr

    # Sequence "seq3" will simulate rain of varying rain fall rate (IMPORTANT: intensity parameter will be omitted). 1st frame will be 30mm/hr rain, 2nd 28mm/hr, ..., Nth 2mm/hr. In general, the ith frame uses i % N simulation (% = modulo)
    settings["sequences"]["seq3"] = {}
    settings["sequences"]["seq3"]["sim_mode"] = "steps"
    settings["sequences"]["seq3"]["sim_steps"] = {"rain_fallrate": [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]}

    return settings

import json
import os

from config.nuscenes.nusc_dataset import NuScenesDataset, NuScenesGANDataset

nusc_dataset, root, tokens = None, None, None

def _sequences(results):
    # how to do "unique" on a list ;)
    unique_sequences = sorted(list(set(nusc_dataset.scene_tokens)))
    if results.sequences:
        is_numeric = results.sequences[0].isnumeric()
        sequences = [int(s) if is_numeric else s for s in results.sequences.split(',')
                     if not is_numeric or int(s) < len(unique_sequences)]
        if is_numeric:
            sequences = [unique_sequences[s] for s in sequences]
        else:
            sequences = sequences
    else:
        sequences = unique_sequences

    return sequences

def _init(results):
    global root, nusc_dataset

    tokens = None
    if results.json_file:
        with open(results.json_file) as f:
            tokens = json.load(f)["sample_data_tokens"]

    is_gan = "gan" in results.dataset
    if is_gan:
        root = results.gan_root
        nusc_dataset = NuScenesGANDataset(version="v1.0-trainval", root=results.dataset_root,
                                          gan_root=results.gan_root, post_fix=results.post_fix,
                                          pretransform_data=False, preload_data=False, only_annotated=False,
                                          specific_tokens=tokens)
    else:
        root = results.dataset_root
        nusc_dataset = NuScenesDataset(version="v1.0-trainval", root=results.dataset_root,
                                       pretransform_data=False, preload_data=False, only_annotated=False,
                                       specific_tokens=tokens)

def resolve_paths(results):
    _init(results)
    results.sequences = _sequences(results)
    assert (len(results.sequences) > 0), "There are no valid sequences folder in the dataset root."

    # Setting images dir of sequence
    results.images = {sequence: [os.path.join(root, filepath) for filepath in nusc_dataset.get_filepaths(sequence, "CAM_FRONT")] for sequence in results.sequences}

    # Setting dir of rain simulator files of all intensities.
    # Using only the simulator files of one sequence because the speed profile does not matter in discrete frames
    sim_path = os.path.join(results.particles, 'nuscenes')
    results.particles = {sequence: {"path": os.path.join(sim_path, sequence), "options": {"preset": ["nuscenes", scene_token, cameras, motions, durations]}} for sequence in results.sequences}

    # since all depth files are more or less in a pile
    results.depth = {sequence: [os.path.join(results.depth_root, os.path.splitext(filename)[0] + ".npy") for filename in nusc_dataset.get_filepaths(sequence, "CAM_FRONT")] for sequence in results.sequences}
    results.calib = {sequence: None for sequence in results.sequences}

    return results

def settings():
    settings = {}
    # settings["cam_hz"] = 10   # Camera Hz (aka FPS)
    # settings["cam_CCD_WH"] = [1242, 375]  # Camera CDD Width and Height (pixels)
    # settings["cam_CCD_pixsize"] = 4.65  # Camera CDD pixel size (micro meters)
    # settings["cam_WH"] = [1242, 375]  # Camera image Width and Height (pixels)
    settings["cam_focal"] = 5.5  # Focal length (mm)
    settings["cam_gain"] = 1.0
    settings["cam_f_number"] = 1.8  # F-Number
    settings["cam_focus_plane"] = 6.0  # Focus plane (meter)
    settings["cam_exposure"] = 5.0  # Camera exposure (ms)

    # Camera extrinsic parameters
    settings["cam_pos"] = [1.5, 1.5, 0.3]  # Camera pos
    settings["cam_lookat"] = [1.5, 1.5, -1.]  # Camera look at vector
    settings["cam_up"] = [0., 1., 0.]  # Camera up vector

    # Sequence-wise settings
    # Note: sequence object and settings are merged, hence any setting can be overwriten sequence-wise
    settings["sequences"] = {}
    # no sequence wise settings

    return settings
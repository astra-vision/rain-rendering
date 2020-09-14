import numpy as np
import os
import re
import sys
import time

from natsort import natsorted


# Return a path OS specific (convert slash to the current OS style)
def path_os_s(path):
    if os.sep == '/':
        return re.sub(r'[/|\\]+', os.sep, path)
    elif os.sep == '\\':
        return re.sub(r'[/|\\]+', re.escape(os.sep), path)
    else:
        raise NotImplemented

def os_listdir(path):
    return natsorted(os.listdir(path))

def print_error(msg):
    print('\n\x1b[2;30;41m[ERROR]\x1b[0m  %s' % msg)


def print_success(msg):
    print('\n\x1b[2;30;42m[SUCCESS]\x1b[0m  %s' % msg)


def print_warning(msg):
    print('\x1b[2;30;43m[WARNING]\x1b[0m  %s' % msg)


def print_progress_bar (iteration, total, prefix='Progress:', suffix='Complete', decimals=2, length=100, fill='#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()


def convert_rgb_to_xyY(array):
    mat = np.array([[0.49000, 0.31000, 0.20000], [0.17697, 0.81240, 0.01063], [0.00000, 0.01000, 0.99000]])
    factor = 0.17697

    XYZ = np.dot(array, mat)/factor
    X = XYZ[..., 0]
    Y = XYZ[..., 1]
    Z = XYZ[..., 2]

    with np.errstate(divide='ignore'):
        x = X/(X + Y + Z)
        y = Y/(X + Y + Z)

    return np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), np.expand_dims(Y, axis=-1)], axis=-1)


def convert_xyY_to_rgb(xyY):

    x = xyY[..., 0]
    y = xyY[..., 1]
    Y = xyY[..., 2]

    X = (Y*x)/y
    Z = (Y * (1 - x - y))/y
    mat = np.array([[0.41847, -0.15866, -0.082835], [-0.091169, 0.25243, 0.015708], [0.0009209, -0.0025498, 0.1786]])

    XYZ = np.concatenate([np.expand_dims(X, axis=-1), np.expand_dims(Y, axis=-1), np.expand_dims(Z, axis=-1)], axis=-1)

    rgb = np.dot(XYZ,mat)

    return rgb


def crop_center(image, height, width):
    crop_x1 = int((image.shape[0] - height) / 2)
    crop_x2 = crop_x1 + height
    crop_y1 = int((image.shape[1] - width) / 2)
    crop_y2 = crop_y1 + width

    image = image[crop_x1:crop_x2, crop_y1:crop_y2]

    return image


# Return the ETA to end of rendering.
def process_eta_str(process_t0, folder_idx, folders_num, folder_t0=None, sim_idx=None, sim_num=None, sim_t0=None,
                    f_idx=None, f_num=None, frame_t0=None, drop_idx=None, drop_num=None):
    frame_progress = drop_idx / drop_num if drop_idx is not None else 0.
    sim_progress = (f_idx + frame_progress) / f_num if f_idx is not None else 0.
    folder_progress = (sim_idx + sim_progress) / sim_num if sim_idx is not None else 0.
    process_progress = (folder_idx + folder_progress) / folders_num

    # S = sequences, F = frames, D = drops
    msg = '          S. {} / {}'.format(sim_idx+1, sim_num)
    if f_idx is not None:
        msg += ', F. {} / {}'.format(f_idx+1, f_num)
    if drop_idx is not None:
        msg += ', D. {} / {}'.format(drop_idx+1, drop_num)
    msg += '     >     MIN remaining time to '

    process_rtime = (1. - process_progress) * (time.time() - process_t0) / process_progress if process_progress else -1
    # Remaining time to total processing
    msg += "End {:02.0f}m".format(process_rtime // 60)

    # Remaining time to complete sequence
    if sim_idx is not None:
        folder_rtime = (1. - folder_progress) * (time.time() - folder_t0) / folder_progress if folder_progress else -1
        msg += ", Seq. {:02.0f}m".format(folder_rtime // 60)

    # Remaining time to weather processing
    if f_idx is not None:
        sim_rtime = (1. - sim_progress) * (time.time() - sim_t0) / sim_progress if sim_progress else -1
        msg += ", Wth. {:02.0f}m".format(sim_rtime // 60)

    # Remaining time to frame processing
    # if drop_idx is not None:
    #     frame_rtime = (1. - frame_progress) * (time.time() - frame_t0) / frame_progress if frame_progress else -1
    #     msg += ", F {:02.0f}m".format(frame_rtime // 60)

    return msg

def hash_(obj, path=False, isclose=-1):
    import hashlib as hl
    import numpy as np
    import re
    if isinstance(obj, dict):
        d = sorted(obj.items())
        return hash_([(k, hash_(v, path=path, isclose=isclose)) for k, v in d].__str__())
    elif isinstance(obj, list):
        return hash_([hash_(v, path=path, isclose=isclose) for v in obj].__str__())
    elif type(obj) in [int, float, bool]:
        if isclose != -1:
            obj = np.round(obj, isclose)
        return str(obj)
    elif isinstance(obj, np.ndarray):
        if isclose != -1 and obj.dtype in [np.float, np.int]:
            obj = np.round(obj, isclose)
            return hash_(obj.tolist(), path=path, isclose=-1)

        return hash_(obj.tolist(), path=path, isclose=isclose)
    elif type(obj) in [str]:
        if path:
            obj = re.sub(r'[/|\\]+', '/', obj)
        return hl.md5(obj.encode()).hexdigest(), obj
    elif isinstance(obj, object):
        if '__str__' in dir(obj):
            if isclose != -1:
                print("WARNING: Conversion to string will prevent truncation")

            return hash_(str(obj), path=path, isclose=isclose)
        d = obj.__dict__
        if "__objclass__" in d:
            del d["__objclass__"]
        return hash_(obj.__dict__, path=path, isclose=isclose)
    else:
        return hl.md5(obj).hexdigest()

def particles_path(path, weather):
    return os.path.join(path, weather["weather"], "{}mm".format(weather["fallrate"]),  '*_camera0.xml')
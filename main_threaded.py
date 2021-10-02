import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np

np.random.seed(0)  # if args.frame_step is the same for each script, the permutation will the same
sys.path.append(os.path.dirname(__file__))


class RainRendering(threading.Thread):
    def __init__(self, args):
        threading.Thread.__init__(self)
        self.args = args
        self.dargs = {args[i]: args[i + 1] for i in range(0, len(args), 2) if i < len(args) - 1}

    def toString(self):
        return " ".join(self.args)

    def run(self):
        pattern = "{}mm_{}_to_{}".format(self.dargs.get("--intensity", "NA"), self.dargs.get("--frame_start", 0),
                                         self.dargs.get("--frame_end", "NA"))
        if self.dargs.get("--frame_step"):
            pattern += "_step_{}".format(self.dargs.get("--frame_step"))

        log_path = os.path.join('automate_log_' + pattern + '.txt')
        err_path = os.path.join('automate_error_' + pattern + '.txt')
        logfile = open(log_path, 'a+')
        errfile = open(err_path, 'a+')
        python_path = sys.executable
        script_path = 'main.py'

        cmd = list()
        cmd.append(python_path)
        cmd.append(script_path)
        for a in self.args:
            cmd.append(a)

        print("Log file: {}".format(log_path))
        child = subprocess.Popen(cmd, stderr=errfile, stdout=logfile)

        child.wait()
        child.kill()

        logfile.close()
        errfile.close()


def check_arg(args):
    # Optimized
    parser = argparse.ArgumentParser(description='Rain renderer method')

    parser.add_argument('--intensity',
                        help='Rain Intensities. List of fall rate comma-separated. E.g.: 1,15,25,50.',
                        type=str,
                        required=True)

    parser.add_argument('--scene_threaded',
                        help='Whether to split scene in multiple threads. While this can considerably speed up generation, it may cause issue if simulation files are not ready.',
                        action='store_true',
                        required=False)

    parser.add_argument('--frame_start',
                        help='frame_start',
                        type=int,
                        default=0,
                        required=False)

    parser.add_argument('--frame_end',
                        help='frame_end',
                        type=int,
                        default=None,
                        required=False)

    parser.add_argument('--frame_step',
                        type=int,
                        required=False)

    parser.add_argument('--frames',
                        type=str,
                        required=False)

    parser.add_argument("--scenes_per_thread",
                        help='Number of scenes in a single thread.',
                        type=int,
                        default=25)

    results, _ = parser.parse_known_args(args)
    results.intensity = np.array([int(i) for i in results.intensity.split(",")])

    print(results)
    return results


if __name__ == "__main__":
    args = check_arg(sys.argv[1:])

    threads = np.zeros((0, 3), object)
    if args.scenes_per_thread > 1:
        max_nb_scenes = 111
        scenes_per_thread = args.scenes_per_thread
    else:
        max_nb_scenes = 1
        scenes_per_thread = 1

    if args.scene_threaded:
        assert args.frame_end or args.frames
        frames_per_thread = 41
        ix = 0

        for frame_start in range(args.frame_start, args.frame_end, frames_per_thread):
            for intensity in args.intensity:
                c = 0
                # modified from range(0, ...)
                for s in range(0, max_nb_scenes, args.scenes_per_thread):
                    _new_args = sys.argv[1:]

                    # Add existing strategy
                    _new_args.append('--conflict_strategy')
                    _new_args.append('skip')
                    _new_args.remove('--scene_threaded')

                    # In multithread, the progress is a bad idea; it could make the log file EXTREMELY big
                    if '-v' in _new_args:
                        _new_args.remove('-v')
                    _new_args.append('--noverbose')

                    # Replace intensity
                    _new_args[_new_args.index('--intensity') + 1] = str(intensity)
                    if not args.frames:
                        # Replace frame_start
                        _new_args[_new_args.index('--frame_start') + 1] = str(frame_start)
                        # Replace frame_end
                        _new_args[_new_args.index('--frame_end') + 1] = str(min(frame_start + frames_per_thread, args.frame_end))
                    elif args.jump and args.frames:
                        _new_args[_new_args.index('--frames') + 1] = ",".join(
                            [str((int(f) + c) % frames_per_thread) for f in _new_args[_new_args.index('--frames') + 1].split(',')])
                        c += args.scenes_per_thread if args.scenes_per_thread else 1

                    if "sequences" in args and args.scenes_per_thread:
                        _new_args.append('--sequences')
                        _new_args.append(",".join([str(j) for j in range(s, np.min((s + args.scenes_per_thread, max_nb_scenes)))]))

                    if "--scenes_per_thread" in _new_args:
                        j = _new_args.index("--scenes_per_thread")
                        del _new_args[j:j + 2]

                    print("Create thread: ", " ".join(_new_args))
                    sim = RainRendering(_new_args)
                    threads = np.vstack([threads, [intensity, ix, sim]])
                    ix += 1
    else:
        for ix, intensity in enumerate(args.intensity):
            _new_args = sys.argv[1:]

            # Add existing strategy
            _new_args.append('--conflict_strategy')
            _new_args.append('skip')

            _new_args.append('--noverbose')

            # Replace intensity
            _new_args[_new_args.index('--intensity') + 1] = str(intensity)

            print("Create thread: ", " ".join(_new_args))
            sim = RainRendering(_new_args)
            threads = np.vstack([threads, [intensity, ix, sim]])

    print("\n---------------")
    print("Note this script does not show real-time output to avoid cumbersome console scrolling. Check ad-hoc logs.")

    # To change, if needed
    max_thread = 10
    while len(threads) > 0:
        thread_not_started_mask = np.array([not t[2]._started.is_set() for t in threads])

        if np.sum(thread_not_started_mask) > 0:
            t = threads[thread_not_started_mask][0, 2]
            print("\n\n>>> START thread: ", t.toString())
            t.start()

        # Wait for an available thread
        print("Wait for threads")
        while np.sum([t[2].isAlive() for t in threads]) >= max_thread:
            time.sleep(2)

        thread_ended_mask = np.array([not t[2].isAlive() and t[2]._started.is_set() for t in threads])
        for t in threads[thread_ended_mask]:
            print("\nThread ended: ", t[2].toString())
        threads = threads[~thread_ended_mask]

        # Wait for all threads if no remaining ones
        if np.sum(np.array([not t[2]._started.is_set() for t in threads])) == 0:
            while np.sum([t[2].isAlive() for t in threads]) != 0:
                time.sleep(2)

            print("All threads completed")

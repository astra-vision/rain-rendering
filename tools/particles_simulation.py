from common import my_utils, db
from tools.simulation import *

force_recompute = False
particles_root = os.path.join('data', 'particles')

_sim_status = None
def process_sequences(sequences, weathers, force_recompute=False):
    simulations = {"path": [], "options": [], "weather": weathers}

    print("Resolve sequences...")
    for s in sequences:
        db_n, seq = s[0], my_utils.path_os_s(s[1])
        sim = db.sim(db_n, seq, os.path.join(particles_root, db_n))

        simulations["path"].append(sim["path"])
        simulations["options"].append(sim["options"])

    print("Run process...")
    return process(simulations, force_recompute=force_recompute)


def process(sim, force_recompute=False):
    path, options, weathers = sim["path"], sim["options"], sim["weather"]
    threads = np.array([], object)

    def print_progress():
        global _sim_status
        threads_active = [t for t in threads if t._started.is_set() and t.isAlive()]
        status = " | ".join(["#{id}: {time:.2f}/{dur:.2f}s".format(id=t.id, time=t.simtime, dur=t.simdur) for t in threads_active])
        if status == _sim_status:
            return

        _sim_status = status
        sys.stdout.write("\r sim. progress: {}".format(_sim_status))

    max_thread = 10
    deactivate_window_mode = True
    redo = force_recompute
    for weather in weathers:
        for i in range(len(path)):
            sim = WeatherSimulation(len(threads), path[i], options[i], weather, redo, deactivate_window_mode)
            threads = np.append(threads, sim)


    while len(threads) > 0:
        thread_not_started_mask = np.array([not t._started.is_set() for t in threads])

        if np.sum(thread_not_started_mask) > 0:
            t = threads[thread_not_started_mask][0]
            print("START thread: ", t.output_dir)
            t.start()
            # to ensure that the seed (which seems to use the time in sec :| ?!) won't be the same
            time.sleep(1.5)

        # Wait for an available thread
        print("Wait for threads")
        while np.sum([t.isAlive() for t in threads]) >= max_thread:
            time.sleep(2)
            print_progress()

        thread_ended_mask = np.array([not t.isAlive() and t._started.is_set() for t in threads])
        for t in threads[thread_ended_mask]:
            print("Thread ended: ", t.output_dir)
        threads = threads[~thread_ended_mask]

        # Wait for all threads if no remaining ones
        if np.sum(np.array([not t._started.is_set() for t in threads])) == 0:
            while np.sum([t.isAlive() for t in threads]) != 0:
                time.sleep(2)
                print_progress()

            print("All threads completed")


if __name__ == "__main__":
    sequences = []

    # Pairs of dataset, sequence to simulate data for
    sequences.append(["kitti", "data_object"])
    sequences.append(["kitti", "raw_data/2011_09_26/2011_09_26_drive_0032_sync"])
    sequences.append(["kitti", "raw_data/2011_09_26/2011_09_26_drive_0056_sync"])
    # sequences.append(["kitti", "raw_data/2011_09_26/2011_09_26_drive_0071_sync"])
    # sequences.append(["kitti", "raw_data/2011_09_26/2011_09_26_drive_0117_sync"])
    sequences.append(["cityscapes", "leftImg8bit"])

    # Weathers to simulate
    weathers = []
    for fallrate in [1, 2, 3, 4, 5, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]:
        weathers.append({"weather": "rain", "fallrate": fallrate})

    process_sequences(sequences, weathers, force_recompute=force_recompute)
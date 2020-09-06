import json
import os
import platform
import signal
import sys
import threading
import time

import numpy as np
from pexpect.exceptions import ExceptionPexpect
from pexpect.popen_spawn import PopenSpawn


class logwriter():
    def __init__(self, outbuffer):
        self._outbuffer = outbuffer

    def write(self, str):
        self._outbuffer.write(str.decode('ascii'))

    def flush(self):
        self._outbuffer.flush()


sequence_code = {}
sequence_code["kitti"] = "80"
sequence_code["cityscape"] = "90"

assert sys.platform[:3] in ['win', 'lin'], "Only windows x64 and linux x64 are supported for the particle simulator"
assert platform.architecture()[0] == '64bit', "Only 64 bits OS are supported for the particle simulator"

bin_platform = sys.platform[:3]
bin_bitness = 'x64'

stats_start_time = 5.0  # Stats starting time (s)

class WeatherSimulation(threading.Thread):
    def __init__(self, id, path, options, weather, redo=False, deactivate_window_mode=True,
                 bin_folder=os.path.join(os.getcwd(), "3rdparty", "weather-particle-simulator", "{}_{}".format(bin_platform, bin_bitness))):
        threading.Thread.__init__(self)

        self.id = id
        self.simtime = 0.  # Time (s)
        self.simdur = 0.  # Duration (s)

        self.options = options
        if "preset" in self.options:
            self.preset = self.options["preset"]
        else:
            self.preset = None

        self.path = path
        self.weather = weather
        self.deactivate_window_mode = deactivate_window_mode
        self.redo = redo
        self.bin_folder = bin_folder
        self.output_dir = os.path.join(self.path, weather["weather"], "{}mm".format(weather["fallrate"]))
        print("Create thread", self.output_dir)

        self.assert_validity()

    def assert_validity(self):
        if self.preset is not None:
            assert self.preset[0] != "kitti" or self.preset[1] in ["0000", "0032", "0056", "0071", "0117"], "Kitti preset is invalid"
            assert self.preset[0] != "cityscapes" or self.preset[1] in ["0000"], "Cityscapes preset is invalid"

    def _print(self, *argv):
        print("\r #{}  ".format(self.id), end='')
        print.__call__(*argv)
    
    def interact(self, wait_for, send_str):
        # print('wait for: {}'.format(wait_for))
        self.child.expect(wait_for)
        # print('send: {}'.format(send_str))
        self.child.sendline(send_str.encode('ascii'))

    def interact_step_menu(self, menu):
        self.interact('Steps: What do you want to do \?', menu)

    def set_sim_steps_times(self, start, dur, last):
        self.interact_step_menu('2')
        self.interact('Enter new duration', '{}'.format(start))

        self.interact_step_menu('3')
        self.interact('Enter new duration', '{}'.format(dur))

        self.interact_step_menu('4')
        self.interact('Enter new duration', '{}'.format(last))

    def set_sim_steps_camera_focal(self, values):
        self.interact_step_menu('12')
        self.interact('What do you want to do \?', '{}'.format(3))
        self.interact('Separator', '{}'.format(";"))
        self.interact('Enter all steps values', '{}'.format(";".join(["{}".format(v) for v in values])))
        self.interact('Continue \?', '{}'.format("y"))

    def set_sim_steps_camera_exposure(self, values):
        self.interact_step_menu('13')
        self.interact('What do you want to do \?', '{}'.format(3))
        self.interact('Separator', '{}'.format(";"))
        self.interact('Enter all steps values', '{}'.format(";".join(["{}".format(v) for v in values])))
        self.interact('Continue \?', '{}'.format("y"))

    def set_sim_steps_camera_motion(self, values):
        self.interact_step_menu('18')
        self.interact('What do you want to do \?', '{}'.format(3))
        self.interact('Separator', '{}'.format(";"))
        self.interact('Enter all steps values', '{}'.format(";".join(["{}".format(v) for v in values])))
        self.interact('Continue \?', '{}'.format("y"))

    def set_sim_steps_rain_fallrate(self, values):
        self.interact_step_menu('41')
        self.interact('What do you want to do \?', '{}'.format(3))
        self.interact('Separator', '{}'.format(";"))
        self.interact('Enter all steps values', '{}'.format(";".join(["{}".format(v) for v in values])))
        self.interact('Continue \?', '{}'.format("y"))

    def interact_main_menu(self, menu):
        self.interact('What do you want to do \?', menu)

    def set_sim_Duration(self, val):
        self.interact_main_menu('6')
        self.interact('Enter new duration', '{}'.format(val))

    def set_sim_Hz(self, val):
        self.interact_main_menu('7')
        self.interact('Enter new frequency', '{}'.format(val))

    def set_sim_ParticlesDetectionLatencyFrames(self, val):
        self.interact_main_menu('61')
        self.interact('Enter new particles detection latency', '{}'.format(val))

    def set_sim_ParticlesDetectionErrorMargin(self, val):
        self.interact_main_menu('62')
        self.interact('Enter new particles detection error', '{}'.format(val))

    def set_sim_Camera0_Hz(self, val):
        self.interact_main_menu('10')
        self.interact('Enter new frequency', '{}'.format(val))

    def set_sim_Camera0_ViewMatrixIC(self, pos, lookat, up):
        self.interact_main_menu('15')
        self.interact('Enter new IC pos x', '{}'.format(pos[0]))
        self.interact('Enter new IC pos y', '{}'.format(pos[1]))
        self.interact('Enter new IC pos z', '{}'.format(pos[2]))

        self.interact('Enter new IC lookat x', '{}'.format(lookat[0]))
        self.interact('Enter new IC lookat y', '{}'.format(lookat[1]))
        self.interact('Enter new IC lookat z', '{}'.format(lookat[2]))

        self.interact('Enter new IC up x', '{}'.format(up[0]))
        self.interact('Enter new IC up y', '{}'.format(up[1]))
        self.interact('Enter new IC up z', '{}'.format(up[2]))

    def set_sim_Camera0_CcdSpecs(self, width, height, pixsize):
        self.interact_main_menu('11')
        self.interact('Camera 0 CCD pxl size', '{}'.format(pixsize))
        self.interact('Camera 0 CCD width', '{}'.format(width))
        self.interact('Camera 0 CCD height', '{}'.format(height))

    def set_sim_Camera0_Focal(self, val):
        self.interact_main_menu('12')
        self.interact('Enter new focal', '{}'.format(val))

    def set_sim_Camera0_Resolution(self, width, height):
        self.interact_main_menu('14')
        self.interact('Camera 0 Resolution WIDTH', '{}'.format(width))
        self.interact('Camera 0 Resolution HEIGHT', '{}'.format(height))

    def set_sim_Camera0_ExposureTime(self, val):
        self.interact_main_menu('13')
        self.interact('Enter new exposure time', '{}'.format(val))

    def set_sim_Camera0_VisibilityMappingAuto(self):
        self.interact_main_menu('17')
        self.interact('Enter new visibility mapping MIN', '{}'.format(0))
        self.interact('Enter new visibility mapping MAX', '{}'.format(0))

    def set_sim_Camera0_MotionSpeedIC(self, val):
        self.interact_main_menu('18')
        self.interact('Enter new initial motion speed', '{}'.format(val))

    def set_sim_Projector0_Hz(self, val):
        self.interact_main_menu('21')
        assert(val == "AHL_HZ_MAX")
        # No need to do anything

    def set_sim_Projector0_Res(self, width, height):
        self.interact_main_menu('22')
        # No need to do anything

    def set_sim_Projector0_MinPixelOverlay(self, val):
        self.interact_main_menu('24')
        self.interact('Enter new minimum pixel overlay', '{}'.format(val))

    def set_sim_Projector0_DbgSaveLightmap(self, val):
        while True:
            self.interact_main_menu('28')
            obs = self.child.expect(["Projector 0 save light maps \(OFF\)", "Projector 0 save light maps \(ON\)"])
            if (val and obs == 1) or (not val and obs == 0):
                break

    def set_sim_Stats_Active(self, val):
        while True:
            self.interact_main_menu('70')
            obs = self.child.expect(["Output simulation stats \(OFF\)", "Output simulation stats \(ON\)"])
            if (val and obs == 1) or (not val and obs == 0):
                break

    def set_sim_Stats_StatsLevel(self, val):
        while True:
            self.interact_main_menu('72')
            obs = self.child.expect(["Stats level \(HIERARCHY\)", "Stats level \(NO HIERARCHY\)"])
            if (val == "HIERARCHY" and obs == 0) or (val == "NO HIERARCHY" and obs == 1):
                break

    def set_sim_Stats_StartTime(self, val):
        self.interact_main_menu('71')
        self.interact('Enter start time', '{}'.format(val))

    def apply_options(self):
        # Series of hacks to avoid warning due to incompatible parameters
        self.set_sim_Camera0_CcdSpecs(99999, 99999, 1.0)

        # Apply settings
        self.set_sim_Duration(stats_start_time * 1000 + self.options["sim_duration"] * 1000)
        self.set_sim_Hz(str(self.options["sim_hz"]))
        self.set_sim_ParticlesDetectionLatencyFrames(0)
        self.set_sim_ParticlesDetectionErrorMargin(0)

        # Camera parameters
        self.set_sim_Camera0_Hz(self.options["cam_hz"])
        self.set_sim_Camera0_ViewMatrixIC(self.options["cam_pos"], self.options["cam_lookat"], self.options["cam_up"])
        self.set_sim_Camera0_Resolution(self.options["cam_WH"][0], self.options["cam_WH"][1])  # Order matters
        self.set_sim_Camera0_CcdSpecs(self.options["cam_CCD_WH"][0], self.options["cam_CCD_WH"][1], self.options["cam_CCD_pixsize"])  # Order matters
        self.set_sim_Camera0_Focal(self.options["cam_focal"])
        self.set_sim_Camera0_ExposureTime(self.options["cam_exposure"])
        self.set_sim_Camera0_VisibilityMappingAuto()
        # self.set_sim_Camera0_motionSpeedIC = 30

        # Projector parameters
        self.set_sim_Projector0_Hz("AHL_HZ_MAX")
        self.set_sim_Projector0_Res(self.options["cam_WH"][0], self.options["cam_WH"][1])
        self.set_sim_Projector0_MinPixelOverlay(50)
        self.set_sim_Projector0_DbgSaveLightmap(True)

        # Debug
        # parameters
        # self.set_sim_Render()->Active(false)
        # self.set_sim_Render()->Hz(200)
        # self.set_sim_Render()->DbgCameraManip(true)
        # self.set_sim_Render()->DbgCameraManipViewMatrix(self.set_sim_Camera0_Pos(), self.set_sim_Camera0_Lookat(), self.set_sim_Camera(
        #     0)->Up())

        self.set_sim_Stats_Active(True)
        self.set_sim_Stats_StartTime("{}".format(stats_start_time*1000))
        self.set_sim_Stats_StatsLevel("HIERARCHY")

    def run(self):
        weather = self.weather

        os.makedirs(self.output_dir, exist_ok=True)

        if not self.redo:
            files = os.listdir(self.output_dir)
            results_computed = np.any(["camera0.xml" in f for f in files])
            if results_computed:
                self._print("Simulation file exits {}, next!".format(self.output_dir))
                return

        # Save options as json dumps
        try:
            fp = open(os.path.join(self.output_dir, "sim_options.json"), 'w')
            options_ = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in self.options.items()}  # Convert to native types
            if "sim_steps" in options_:
                options_["sim_steps"] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in options_["sim_steps"].items()}
            json.dump(options_, fp)
            fp.close()
        except Exception as e:
            print(e)
            print("Failed saving JSON {}... Not crucial, continuing")

        log_path = os.path.join(self.output_dir, 'automate_log.txt')
        log_fp = open(log_path, 'a+')
        # self._print(self.output_dir)
        self.child = PopenSpawn(os.path.join(self.bin_folder, 'AHLSimulation'), cwd=self.output_dir, logfile=logwriter(log_fp))

        try:
            self._print("In main menu")

            self.interact_main_menu('9')
            self.interact('Set the seed for random generator', '0')

            if self.preset is None:
                self._print("Apply options:", self.options)
                self.apply_options()
            else:
                self._print("Apply preset (ignoring options):", self.preset[:2])
                if self.preset[0] in sequence_code:
                    self.interact_main_menu('99')
                    time.sleep(0.5)

                    self._print("Setting system")
                    self.child.expect('Which system to run ?')
                    seq_code = sequence_code[self.preset[0]]+self.preset[1]
                    self._print('		System code: ', seq_code)
                    self.child.sendline(seq_code.encode('ascii'))
                elif "nuscenes" in self.preset[0].lower():
                    self.interact_main_menu('99')
                    time.sleep(0.5)

                    self._print("Setting system")
                    self.child.expect('Which system to run ?')
                    if "2Hz" in self.preset[0]:
                        seq_code = '1000'
                    else:
                        seq_code = '100'
                    self._print('		System code: ', seq_code)
                    self.child.sendline(seq_code.encode('ascii'))
                else:
                    raise NotImplementedError("No settings for this set {}".format(self.preset[0]))

            # Deactivate windows AND save light map option
            if self.deactivate_window_mode:
                self._print("In main menu")
                self.interact_main_menu('28')
                time.sleep(0.5)
                self._print("	Save light map")

            # Deactivate rain particles
            self._print("Deactivating rain particles")
            self.interact_main_menu('410')
            self.child.expect('410. Rain \(OFF\)')

            if weather["weather"] == "rain":
                self._print("Activating rain particles")
                # Activate rain particles
                self.interact_main_menu('410')
                self.child.expect('410. Rain \(ON\)')

                self._print("Setting rain fallrate")
                # Set rain fallrate
                self.interact_main_menu('414')
                self.child.expect('Enter new Rain fall rate')
                code = str(weather["fallrate"])
                self.child.sendline(code.encode('ascii'))

            self._print("In main menu")

            _steps_menu = False
            if self.preset is None:
                assert self.options["sim_mode"] in ["normal", "steps"]

                # Actions in main menu
                if self.options["sim_mode"] == "steps":
                    assert ("sim_steps" in self.options and type(self.options["sim_steps"]) == dict and len(self.options["sim_steps"].keys()) != 0), "[ERROR]: sim_steps must be provided and be a non-empty dictionary."
                    assert np.all([type(v) == list or type(v) == np.ndarray for k,v in self.options["sim_steps"].items()]), "[ERROR]: values in sim_steps should all be list/arrays"

                    # Apply initial cam speed
                    if "cam_motion" in self.options["sim_steps"]:
                        speedIC = self.options["sim_steps"]["cam_motion"][0]
                        self.set_sim_Camera0_MotionSpeedIC(speedIC)

                # Actions in step menu
                if self.options["sim_mode"] == "steps":
                    # Enter step menu
                    self.interact_main_menu('102')
                    _steps_menu = True

                    # Set step durations
                    step_dur = 1. / self.options["cam_hz"]
                    self.set_sim_steps_times(stats_start_time*1000, step_dur*1000, step_dur*1000)

                    # Attempt to compute the total simulation time
                    max_steps = max([len(v) for k,v in self.options["sim_steps"].items()])
                    self.simdur = stats_start_time + (1+max_steps)*step_dur

                    if "cam_motion" in self.options["sim_steps"]:
                        self._print(" Apply {} steps cam motion: {}".format(len(self.options["sim_steps"]["cam_motion"]), ";".join(["{}".format(v) for v in self.options["sim_steps"]["cam_motion"]])))
                        self.set_sim_steps_camera_motion(self.options["sim_steps"]["cam_motion"])
                    if "cam_focal" in self.options["sim_steps"]:
                        self._print(" Apply {} steps cam focal: {}".format(len(self.options["sim_steps"]["cam_focal"]), ";".join(["{}".format(v) for v in self.options["sim_steps"]["cam_focal"]])))
                        self.set_sim_steps_camera_focal(self.options["sim_steps"]["cam_focal"])
                    if "cam_exposure" in self.options["sim_steps"]:
                        self._print(" Apply {} steps cam exposure: {}".format(len(self.options["sim_steps"]["cam_exposure"]), ";".join(["{}".format(v) for v in self.options["sim_steps"]["cam_exposure"]])))
                        self.set_sim_steps_camera_exposure(self.options["sim_steps"]["cam_exposure"])
                    if "rain_fallrate" in self.options["sim_steps"]:
                        self._print(" Apply {} steps rain fallrate: {}".format(len(self.options["sim_steps"]["rain_fallrate"]), ";".join(["{}".format(v) for v in self.options["sim_steps"]["rain_fallrate"]])))
                        self.set_sim_steps_rain_fallrate(self.options["sim_steps"]["rain_fallrate"])

                    if self.options["sim_mode"] == "normal":
                        self.simdur = stats_start_time + self.options["sim_duration"]
            else:
                # Enter step menu
                self.interact_main_menu('102')
                _steps_menu = True

            # self._print("Going to step menu")
            # self.interact_main_menu('102')

            # if "nuscenes" in preset[0].lower():
            #     self.child.expect('Steps: What do you want to do \?')
            #     self._print("In Step menu")
            #     self.child.sendline('18'.encode('ascii'))
            #     self._print("Camera 0 motion speed choices")
            #     self.child.expect("What do you want to do")
            #     self.child.sendline('3'.encode('ascii'))
            #     self._print("		Camera 0 motion speed choices (3 -> all at once)")
            #
            #     speed = np.linalg.norm(np.array(preset[3]), axis=1) / 1000 * 3600 / (np.array(preset[4]) * 1e-6)
            #     self.child.expect("Separator")
            #     self.child.sendline(';'.encode('ascii'))
            #     self.child.expect("Enter all steps values")
            #     self._print("		Camera 0 motion speeds (min, max): ({}, {})".format(np.min(speed), np.max(speed)))
            #     self.child.sendline(';'.join([str(s) for s in speed.tolist()]).encode('ascii'))
            #     self.child.expect("Continue")
            #     self.child.sendline('y'.encode('ascii'))

            if _steps_menu:
                self._print("In Step menu")

            self._print("Starting simulation")
            self.interact_step_menu('1')

            index = None
            while index != 1:
                index = self.child.expect(['[0-9]+:[0-9]+:[0-9]+\.[0-9]* +\(p#[0-9]*\)', '\[Simulation stopped\]'], timeout=None)
                if index == 0:
                    self.simtime += 0.5  # Each time a time is displayed the simulation advanced of 0.5 seconds

            self._print("Simulation stopped")
            time.sleep(5.)  # Wait for the "Press any key to continue"
            self.child.sendline(b'\n')

            if _steps_menu:
                self.child.expect('Steps: What do you want to do \?')
                self._print("In Step menu")
                self._print("Going to main menu")
                self.child.sendline('0'.encode('ascii'))
                _steps_menu = False

            self._print("In main menu")
            self._print("Stopping process")
            self.child.expect('What do you want to do \?')
            self.child.sendline('0'.encode('ascii'))

            self.child.expect('Press any key to continue . . .')
            self.child.sendline(b'\n')
            self.child.expect('Press any key to continue . . .')
            self.child.sendline(b'\n')
        except ExceptionPexpect as e:
            self._print("ERROR occured. Check the content of the log file to solve: {}".format(log_path))
            self._print(e)
            self._print("If starting python from an IDE, check LD_LIBRARY_PATH environment is accessible (https://youtrack.jetbrains.com/issue/PY-29580)")
            self.child.kill(signal.SIGINT)
            exit()
        except Exception as e:
            # Might happens at the end of the simulation, if the process closes slightly before last key strike
            self._print(e)
            pass

        # Kill process in case it's not dead (cruel world)
        try:
            self.child.wait()
            self.child.kill(signal.SIGINT)
        except Exception:
            pass

        log_fp.close()

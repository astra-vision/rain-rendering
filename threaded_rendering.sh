#!/usr/bin/env bash

# Sample code to run threaded rendering
python main_threaded.py --intensity 5,25,50 --frame_start 0 --frame_end 41 --scenes_per_thread 10 --dataset nuscenes --dataset_root /data/ --rain /data/ahl/ -sd /data/GargAndNayar/env_light_database --depth /data/depths/ --json_file ../weather_gan_hybrid/splits/v1.0-trainval_both_nnight_nrain_test.json --noise_scale 0.75 --opacity_attenuation 0.6
# python main_threaded.py --intensity 100 --frame_start 0 --frame_end 41 --scenes_per_thread 10 --dataset nuscenes_gan --dataset_root /data/ --gan_root /data/gan/nuscenes/samples/CAM_FRONT --rain /data/ahl/ -sd /data/rainstreakdb/env_light_database --depth /data/depths/ --json_file ../weather_gan_hybrid/splits/v1.0-trainval_both_nnight_nrain_test.json --noise_scale 0.75 --opacity_attenuation 0.6


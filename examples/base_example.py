#!/usr/bin/python3

# Author: Kilian Holzapfel <kilian.holzapfel@tum.de>

import sdaq
import h5py
import time
import numpy as np
import os
import logging

# disable logger
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('schedule').setLevel(logging.WARNING)


# define getter functions
def getter():
    return [np.random.random()]


def getter_2():
    return [np.random.random(), np.random.random((1, 2, 2))]


# set up Jobs
job_list = [sdaq.DAQJob('job_1', 'table_0', getter),
            sdaq.DAQJob('job_2', ['1d_array', 'nd_array'], getter_2, shape=[(0,), (0, 2, 2)], read_period=.2)]
# set up Daemon
daq_daemon = sdaq.DAQDaemon(job_list)

# start the Daemon, wait, and stop it
daq_daemon.start(attrs={'measurement': 'test', 'run_id': 12})  # attrs are added as root attributes of the hdf5 file
time.sleep(3)
daq_daemon.stop()

# check the file
with h5py.File(daq_daemon.file_name, mode='r') as f:
    print(f'{f.filename}')
    for i in f:
        print(f'---> {i}')
        for j in f.get(i):
            data = f.get(f'{i}/{j}')[:]
            print(f'{i}/{j} \t- shape: {data.shape}')
            if j == 'time':
                print(f'{data}')
        print()

# delete the file
os.remove(daq_daemon.file_name)

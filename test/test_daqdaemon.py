import glob
import os
import time
from unittest import TestCase

import h5py
import numpy as np

from src.sdaq import DAQJob, DAQDaemon


class TestDAQDaemonInit(TestCase):
    def test__init_default(self):
        pass


class TestDAQJobFunctions(TestCase):
    def test_default(self):
        pass


class TestDAQJob(TestCase):
    def setUp(self):
        self.file_format_str = "DAQ_{date_from}_{date_to}.hdf5"
        self.file_search_str = self.file_format_str.replace('{date_from}', '*').replace('{date_to}', '*')
        # remove generated files
        for i in glob.glob(self.file_search_str):
            os.remove(i)

    def tearDown(self):
        pass

    def test_basic(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'read_period': .01,
                         'compress': False}

        # ---- BASIC functionality no fails ----
        # test basic functionality
        settings_dict.update({'getter': lambda: [np.random.random((1, *j[1:])) for j in settings_dict['shape']]})
        job = DAQJob('test_job_1', **settings_dict)
        job2 = DAQJob('test_job_2', 'table_0', getter=lambda: [np.random.random()], read_period=.02)

        rollover_interval = 1.
        daemon = DAQDaemon(job_list=[job, job2], name='test_daemon',
                           write_period=0.2, file_format_str=self.file_format_str,
                           rollover_interval_dict={'hours': rollover_interval})

        total_time = rollover_interval * 2.
        # total_time = daemon.write_period * (loops - .5)

        daemon.start({'run': 'test'})
        time.sleep(total_time)  # loops + a half loop
        print('stop', time.time())
        print(daemon.stop())

        self.assertEqual([job.buffer_position, job2.buffer_position], [0, 0],
                         msg='There are items left in the jobs buffer')
        self.assertTrue(os.path.exists(daemon.file_name))

        file = glob.glob(self.file_search_str)
        for i in np.sort(file):
            with h5py.File(i, mode='r') as f:
                print(i, dict(f.attrs))

        with h5py.File(daemon.file_name, mode='r') as f:
            # Job
            length = total_time/job.read_period
            self.assertAlmostEqual(length, f.get('test_job_1/time').shape[0],
                                   delta=length*.2)  # 20% derivation

            for i in f.get('test_job_1').keys():
                self.assertIn(i, job.label)

            # Job2
            length = total_time/job2.read_period
            self.assertAlmostEqual(length, f.get('test_job_2/time').shape[0],
                                   delta=length * .2)  # 20% derivation
            for i in f.get('test_job_2').keys():
                self.assertIn(i, job2.label)

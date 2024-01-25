import logging
import os
import subprocess
import time
from unittest import TestCase

import numpy as np

from src.sdaq import DAQJob, DAQJobExtern


class TestDAQJobInit(TestCase):
    def test__init_default(self):
        job = DAQJob('test', label='table_1', getter=lambda: [np.random.random()])
        self.assertEqual(job.group, 'test')
        self.assertEqual(job.label, ['time', 'table_1'])
        self.assertEqual(job.dtype_list, [np.float64, np.float32])
        self.assertEqual(job.shape_list, [(0,), (0,)])

    def test__init_1(self):
        # create settings
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)]}
        # create getter from shape
        settings_dict.update({'getter': lambda: [np.random.random(i[1:]) for i in settings_dict['shape']]})

        # test label + dtype + shape
        job = DAQJob('test', **settings_dict)
        self.assertEqual(job.label[1:], settings_dict['label'])
        self.assertEqual(job.dtype_list[1:], settings_dict['dtype'])
        self.assertEqual(job.shape_list[1:], settings_dict['shape'])

        # test label + dtype
        settings_dict.pop('shape')
        job = DAQJob('test', **settings_dict)
        self.assertEqual(job.label[1:], settings_dict['label'])
        self.assertEqual(job.dtype_list[1:], settings_dict['dtype'])
        self.assertEqual(job.shape_list[1:], [(0,)] * len(settings_dict['label']))

        # test label only
        settings_dict.pop('dtype')
        job = DAQJob('test', **settings_dict)
        self.assertEqual(job.label[1:], settings_dict['label'])
        self.assertEqual(job.dtype_list[1:], [np.float32] * len(settings_dict['label']))
        self.assertEqual(job.shape_list[1:], [(0,)] * len(settings_dict['label']))

        # test label + shape
        settings_dict.update({'shape': [(0, 2), (0, 2, 2), (0,)]})
        job = DAQJob('test', **settings_dict)
        self.assertEqual(job.label[1:], settings_dict['label'])
        self.assertEqual(job.dtype_list[1:], [np.float32] * len(settings_dict['label']))
        self.assertEqual(job.shape_list[1:], settings_dict['shape'])

        # test label + wrong shape
        settings_dict.update({'shape': [(0, 2, 2), (0,)]})
        self.assertRaises(ValueError, DAQJob, 'test', **settings_dict)
        settings_dict.update({'shape': [(0, 2, 2), 0, (0,)]})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)
        settings_dict.update({'shape': 123})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)
        settings_dict.pop('shape')

        # test label + wrong dtype
        settings_dict.update({'dtype': [int]})
        self.assertRaises(ValueError, DAQJob, 'test', **settings_dict)
        settings_dict.update({'dtype': [int, 1, int]})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)
        settings_dict.update({'dtype': 1})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)
        settings_dict.pop('dtype')

        # test wrong getter
        settings_dict.update({'getter': [int]})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)
        settings_dict.update({'getter': 1.})
        self.assertRaises(TypeError, DAQJob, 'test', **settings_dict)

    # def test__read_values(self):
    #     self.fail()
    #
    # def test__thread_read(self):
    #     self.fail()


class TestDAQJobFunctions(TestCase):
    def test_get_data(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)]}

        # ---- BASIC functionality no fails ----
        # test basic functionality
        settings_dict.update({'getter': lambda: [np.random.random(i[1:]) for i in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)
        self.assertIsNotNone(job._get_data())

        # test basic functionality + provide_time=True
        settings_dict.update({'getter': lambda: [1., *[np.random.random(i[1:]) for i in settings_dict['shape']]],
                              'provide_time': True})
        job = DAQJob('test', **settings_dict)
        self.assertIsNotNone(job._get_data())
        settings_dict.pop('provide_time')

        # test skipp measurement, when getter returns None
        settings_dict.update({'getter': lambda: None})
        job = DAQJob('test', **settings_dict)
        self.assertIsNone(job._get_data())

        # ---- FAILS ----
        # ---- getter fails ----
        # test when getter() fails or raise
        settings_dict.update({'getter': lambda: 0 / 0})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(RuntimeError, job._get_data)

        # ---- getter returns wrong values ----
        # test len(label) > 1 + getter doesn't return iterable
        settings_dict.update({'getter': lambda: np.random.random()})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(TypeError, job._get_data)

        # test len(label) > 1 + len(getter()) != len(label)
        settings_dict.update({'getter': lambda: [np.random.random()]})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._get_data)

        # test len(label) > 1 + len(getter()) != len(label)
        settings_dict.update({'getter': lambda: np.random.random(len(settings_dict['label']) - 1)})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._get_data)

        # ---- 'provide_time' = True ----
        # test if getter() fails
        settings_dict.update({'getter': lambda: None, 'provide_time': True})
        job = DAQJob('test', **settings_dict)
        self.assertIsNone(job._get_data())

        # test fail if getter doesn't return time but provide_time = True
        settings_dict.update({'getter': lambda: [np.random.random(i[1:]) for i in settings_dict['shape']],
                              'provide_time': True})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._get_data)
        settings_dict.pop('provide_time')

    def test_append_buffer(self, compress=False):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'compress': compress,
                         'initial_buffer_length': 10}
        # add the 100* to get values from 0->100
        settings_dict.update({'getter': lambda: [100 * np.random.random(j[1:]) for j in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)

        # ---- BASIC functionality no fails ----
        # check if append change position
        data = []

        # check initial buffer length
        self.assertEqual(job.buffer[0].shape[0], settings_dict['initial_buffer_length'])
        for i in range(50):
            data = job._get_data()  # generate data
            job._append_buffer(data)  # append it
            self.assertEqual(job.buffer_position, i + 1)  # check it buffer position

            # test resizing of array, resizing happens before appending the data -> job.buffer_position-1
            if settings_dict['initial_buffer_length'] == job.buffer_position - 1:
                self.assertEqual(job.buffer[0].shape[0], 2 * settings_dict['initial_buffer_length'])

        # check if data is the last item
        # convert it to the defined dtype
        data_dtype = [np.array([data_i], dtype=dtype_i)[-1:] for data_i, dtype_i in zip(data, job.dtype_list)]
        buffer_last = [item_i[-1:] for item_i in job.peek()]  # get the last item from the buffer
        self.assertTrue(any([i == j for i, j in zip(data_dtype, buffer_last)]))

        # ---- FAILS ----
        # check when getter returns wrong shape
        settings_dict.update({'getter': lambda: [100 * np.random.random((2, *j[1:])) for j in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._append_buffer, job._get_data())

        # check when getter returns wrong type, None can't be interpreted as int only as an float
        def getter():
            return [np.nan * np.random.random((2, *j[1:])) for j in settings_dict['shape']]

        settings_dict.update({'getter': getter})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._append_buffer, job._get_data())

    def test_extend_buffer(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'extend_mode': True,
                         'initial_buffer_length': 10}
        data_length = 10
        # add the 100* to get values from 0->100
        settings_dict.update(
            {'getter': lambda: [np.arange(data_length, dtype=np.float64),  # time
                                *[100 * np.random.random((data_length, *j[1:])) for j in settings_dict['shape']]]})

        job = DAQJob('test', **settings_dict)
        # check initial buffer length
        self.assertEqual(job.buffer[0].shape[0], settings_dict['initial_buffer_length'])

        # ---- BASIC functionality no fails ----
        # check if append change position
        data = []

        for i in range(50):
            data = job._get_data()  # generate data
            job._extend_buffer(data)  # extend it
            self.assertEqual(job.buffer_position, (i + 1) * data_length)  # check it buffer position

            # test resizing of array, resizing happens before appending the data -> job.buffer_position-1
            if settings_dict['initial_buffer_length'] == job.buffer_position - 1:
                self.assertEqual(job.buffer[0].shape[0], 2 * settings_dict['initial_buffer_length'])

        # check if data is the last item
        # convert it to the defined dtype
        data_dtype = [np.array([data_i[-1]], dtype=dtype_i)[-1:] for data_i, dtype_i in zip(data, job.dtype_list)]
        buffer_last = [item_i[-1:] for item_i in job.peek()]  # get the last item from the buffer
        self.assertTrue(any([i == j for i, j in zip(data_dtype, buffer_last)]))

        # ---- FAILS ----
        # check when getter returns wrong shape, i.e. the time has data_length*2
        settings_dict.update(
            {'getter': lambda: [np.arange(data_length * 2, dtype=np.float64),  # time
                                *[100 * np.random.random((data_length, *j[1:])) for j in settings_dict['shape']]]})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._extend_buffer, job._get_data())

        # check when getter returns wrong type, None can't be interpreted as int only as an float
        def getter():
            return [np.arange(data_length, dtype=np.float64),  # time
                    *[np.nan * np.random.random((2, *j[1:])) for j in settings_dict['shape']]]

        settings_dict.update({'getter': getter})
        job = DAQJob('test', **settings_dict)
        self.assertRaises(ValueError, job._extend_buffer, job._get_data())

    def test_append_buffer_compress(self):
        # ---- BASIC functionality no fails ----
        # test when changing data
        self.test_append_buffer(compress=True)

        # test when non changing data
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'compress': True}
        # add the 100* to get values from 0->100
        settings_dict.update({'getter': lambda: [100 * np.random.random(j[1:]) for j in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)

        # ---- BASIC functionality no fails ----
        # check initial buffer length
        data = job._get_data()  # generate data
        data[0] = 1.1  # update time for better readability of values
        job._append_buffer(data)  # append it
        self.assertFalse(job._compress_last_data)
        self.assertEqual(job.buffer_position, 1)  # check it buffer position
        data[0] += 0.1  # update time
        job._append_buffer(data)  # append it
        self.assertTrue(job._compress_last_data)
        self.assertEqual(job.buffer_position, 2)  # check it buffer position
        for i in range(3):
            data[0] += 0.1  # update time
            job._append_buffer(data)  # append it
            self.assertTrue(job._compress_last_data)
            self.assertEqual(job.buffer_position, 2)  # check it buffer position

        # check if the time is updated, the job.buffer_position to the next item -> -1 it the last
        self.assertTrue(job.buffer[0][job.buffer_position - 1] == data[0])

        # check if appending new data works
        data = job._get_data()
        job._append_buffer(data)
        self.assertEqual(job.buffer_position, 3)
        self.assertFalse(job._compress_last_data)
        # convert it to the defined dtype
        data_dtype = [np.array([data_i], dtype=dtype_i)[-1:] for data_i, dtype_i in zip(data, job.dtype_list)]
        buffer_last = [item_i[:job.buffer_position][-1:] for item_i in job.buffer]  # get the last item from the buffer
        self.assertTrue(any([i == j for i, j in zip(data_dtype, buffer_last)]))

    def test_get_buffer(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'read_period': .01,
                         'compress': True}

        settings_dict.update({'getter': lambda: [np.random.random(s[1:]) for s in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)

        data = job._get_data()  # generate data
        data[0] = 0.9  # update time for better readability of values

        # append some data
        for i in range(3):
            # loop to append some data
            for j in range(30):
                data[0] += 0.1  # update time
                job._append_buffer(data)  # append it

            # get buffer and check it length
            buffer = job.get_buffer()
            if i == 0:
                self.assertEqual(len(buffer[0]), 1)  # should return the starting element
            else:
                self.assertEqual(buffer, [])  # should return nothing

        # append some data
        for i in range(3):
            # loop to append some data
            data[1:] = job._get_data()[1:]
            for j in range(30):
                data[0] += 0.1  # update time
                job._append_buffer(data)  # append it

            # get buffer and check it length
            buffer = job.get_buffer()
            # should return 2 elements: the (1) ending of the data old and (2) starting element of the new data
            # after job._get_data()[1:] -> a bit like [data_old, data_new]
            self.assertEqual(len(buffer[0]), 2)


class TestDAQJob(TestCase):
    def test_basic(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3', 'table_4'],
                         'dtype': [int, int, float, str],
                         'shape': [(0, 2), (0, 2, 2), (0,), (0,)],
                         'read_period': .01,
                         'compress': False}

        # ---- BASIC functionality no fails ----
        # test basic functionality
        loops = 10
        settings_dict.update({'getter': lambda: [np.random.random(i[1:]) for i in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)
        job.start()
        time.sleep(settings_dict['read_period'] * (loops + .5))  # loops + a half loop
        job.stop()

        self.assertAlmostEqual(loops, job.buffer_position, delta=1)  # maybe delta=1

    def test_trigger_read(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'read_period': .01,
                         'compress': False}

        loops = 10
        settings_dict.update({'getter': lambda: [np.random.random(j[1:]) for j in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)
        for i in range(loops):
            job.trigger_read()

        self.assertAlmostEqual(loops, job.buffer_position, delta=1)  # maybe delta=1


class TestDAQJobExtern(TestCase):
    def setUp(self):
        # disable logger
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('schedule').setLevel(logging.WARNING)

        self.process_exec_time = time.time()
        cmd = os.path.join(os.path.dirname(__file__), 'external_job_test_script.py')
        proc = subprocess.Popen(f'{cmd} 10 --loops 1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc.wait()
        stderr = proc.stderr.readlines()
        if stderr:
            raise RuntimeError(stderr)
        proc.stdout.close()
        proc.stderr.close()
        self.process_exec_time = time.time() - self.process_exec_time
        #print(self.process_exec_time)

    def test_basic_non_constant(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3', 'table_4'],
                         # only float for now supported by external_job_test_script.py
                         'dtype': [float, float, float, str],
                         'shape': [(0,), (0,), (0,), (0,)],
                         'read_period': .1 + self.process_exec_time,  # depends on the system and could cause fails
                         'compress': False,
                         # 'read_mode': 'constant',
                         # 'extend_mode': True
                         }

        # ---- BASIC functionality no fails ----
        # test basic functionality
        loops = 2
        if "read_mode" in settings_dict and settings_dict["read_mode"] == "constant":
            loops_process = loops
        else:
            loops_process = 1

        length = np.sum([np.prod([1, *i[1:]]) for i in settings_dict['shape']])
        settings_dict.update({'getter': f'external_job_test_script.py {length} --loops {loops_process} '
                                        f'{"--provide_time" if "provide_time" in settings_dict else ""} '
                                        f'--period {settings_dict["read_period"]}'})
        # print(settings_dict['getter'])
        job = DAQJobExtern('test', **settings_dict)
        # print(job.dtype_structured)

        job.start()
        time.sleep(settings_dict['read_period'] * loops)  # (loops + a half loop) + .5s to start process
        job.stop()

        self.assertAlmostEqual(loops, job.buffer_position, delta=1)  # maybe delta=1
        # print([(i, buffer_i.astype(float)) for i, buffer_i in enumerate(job.peek()[1:])])
        self.assertTrue(np.all([i == buffer_i.astype(float) for i, buffer_i in enumerate(job.peek()[1:])]))

    def test_basic_constant(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3', 'table_4'],
                         # only float for now supported by external_job_test_script.py
                         'dtype': [float, float, float, str],
                         'shape': [(0,), (0,), (0,), (0,)],
                         'read_period': .01,  # depends on the system and could cause fails
                         'compress': False,
                         'read_mode': 'constant',
                         # 'extend_mode': True
                         }

        # ---- BASIC functionality no fails ----
        # test basic functionality
        loops = 10
        if "read_mode" in settings_dict and settings_dict["read_mode"] == "constant":
            loops_process = -1  # loops
        else:
            loops_process = 1

        length = np.sum([np.prod([1, *i[1:]]) for i in settings_dict['shape']])
        settings_dict.update({'getter': f'external_job_test_script.py {length} --loops {loops_process} '
                                        f'{"--provide_time" if "provide_time" in settings_dict else ""} '
                                        f'--period {settings_dict["read_period"]}'})
        job = DAQJobExtern('test', **settings_dict)
        job.start()
        # (loops + a half loop) + .5s to start process
        time.sleep(settings_dict['read_period'] * loops + self.process_exec_time)
        job.stop()

        self.assertGreaterEqual(job.buffer_position, loops // 2, msg=f'job.peek()= {job.peek()}')

    def test_basic_extend(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3', 'table_4'],
                         # only float for now supported by external_job_test_script.py
                         'dtype': [float, float, float, str],
                         'shape': [(0,), (0, 2), (0, 2, 2), (0,)],
                         'read_period': .01,  # depends on the system and could cause fails
                         'compress': False,
                         'read_mode': 'constant',
                         'extend_mode': True
                         }

        # ---- BASIC functionality no fails ----
        # test basic functionality
        loops = 10
        if "read_mode" in settings_dict and settings_dict["read_mode"] == "constant":
            loops_process = loops
        else:
            loops_process = 1
        length = np.sum([np.prod([1, *i[1:]]) for i in settings_dict['shape']])
        settings_dict.update({'getter': f'external_job_test_script.py {length} --loops {loops_process} '
                                        f'--provide_time '
                                        f'--period {settings_dict["read_period"] / 2.}'})
        job = DAQJobExtern('test', **settings_dict)
        job.start()
        # (loops + a half loop) + .5s to start process
        time.sleep(settings_dict['read_period'] * (loops + .5) + self.process_exec_time)
        job.stop()

        self.assertGreaterEqual(job.buffer_position, loops // 2, msg=f'job.peek()= {job.peek()}')

    def test_trigger_read(self):
        settings_dict = {'label': ['table_1', 'table_2', 'table_3'],
                         'dtype': [int, int, float],
                         'shape': [(0, 2), (0, 2, 2), (0,)],
                         'read_period': .01,
                         'compress': False}

        loops = 10
        settings_dict.update({'getter': lambda: [np.random.random(j[1:]) for j in settings_dict['shape']]})
        job = DAQJob('test', **settings_dict)
        for i in range(loops):
            job.trigger_read()

        self.assertAlmostEqual(loops, job.buffer_position, delta=1)  # maybe delta=1

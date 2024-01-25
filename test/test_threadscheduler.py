import time
from unittest import TestCase

from h5daq.src.h5daq.scheduler import ThreadScheduler


class TestThreadScheduler(TestCase):
    def setUp(self):
        # logger = logging.getLogger()
        # logger.setLevel(logging.DEBUG)
        # ch = logging.StreamHandler()
        # ch.setLevel(logging.DEBUG)
        # ch.setFormatter(logging.Formatter('%(asctime)s;%(levelname)7s; %(name)20s - %(message)s'))  # formatter)
        # logger.addHandler(ch)

        self.schedule = ThreadScheduler()

    def test_single_job(self):
        exec_dict = {}

        def job(name=None):
            if name not in exec_dict.keys():
                exec_dict[name] = 1
            else:
                exec_dict[name] += 1

        # total time is ~.4s
        self.schedule.every(.1).seconds.do(job, name='Alice')
        time.sleep(.35)
        self.schedule.stop()

        self.assertEqual(3, exec_dict['Alice'])

    def test_clear(self):
        exec_dict = {}

        def job(name=None):
            if name not in exec_dict.keys():
                exec_dict[name] = 1
            else:
                exec_dict[name] += 1

        # total time is ~.4s
        self.schedule.every(.3).seconds.do(job, name='Bob')
        time.sleep(.1)  # test update of
        self.schedule.every(.1).seconds.do(job, name='Alice')
        time.sleep(.35)
        self.schedule.clear()

        self.assertEqual(3, exec_dict['Alice'])
        self.assertEqual(1, exec_dict['Bob'])

    def test_stop(self):
        exec_dict = {}

        def job(name=None):
            if name not in exec_dict.keys():
                exec_dict[name] = 1
            else:
                exec_dict[name] += 1

        self.schedule.every(.1).seconds.do(job, name='Alice')
        self.schedule.every(.2).seconds.do(job, name='Bob')

        time.sleep(.35)
        self.schedule.stop()

        self.assertEqual(3, exec_dict['Alice'])
        self.assertEqual(1, exec_dict['Bob'])

    def test_start_stop(self):
        exec_dict = {}

        def job(name=None):
            if name not in exec_dict.keys():
                exec_dict[name] = 1
            else:
                exec_dict[name] += 1

        self.schedule.every(.2, start=False).seconds.do(job, name='Bob')
        self.schedule.every(.1, start=False).seconds.do(job, name='Alice')

        self.schedule.start()
        time.sleep(.35)
        self.schedule.stop()

        self.assertEqual(3, exec_dict['Alice'])
        self.assertEqual(1, exec_dict['Bob'])

#!/usr/bin/python3

# Author: Dr. Christian Fruck <cfruck@ph.tum.de>
# Kilian Holzapfel <kilian.holzapfel@tum.de>
import datetime
import logging
import random
import threading
import traceback

from schedule import Scheduler, Job, ScheduleError, ScheduleValueError


class ThreadJob(Job):
    def do(self, job_func, *args, **kwargs):
        """
        Specifies the job_func that should be called every time the
        job runs.

        Any additional arguments are passed on to job_func when
        the job runs.

        :param job_func: The function to be scheduled
        :return: The invoked job instance
        """
        Job.do(self, job_func, *args, **kwargs)  # raise if no 'self.scheduler' available

        self.scheduler.logger.debug(f'Scheduler {self.scheduler.name} new Job: {job_func.__name__}')
        self.scheduler.update_event_wait()  # stops the Event.wait to renew the wait period

        if self.scheduler.thread_should_start:
            self.scheduler.start()  # start the loop if it isn't running
            self.scheduler.thread_should_start = False

        return self

    def _schedule_next_run(self) -> None:
        """
        Copy of the original, replaced randomint with random.uniform to allow floats, too.
        Compute the instant when this job should run next.
        """
        if self.unit not in ("seconds", "minutes", "hours", "days", "weeks"):
            raise ScheduleValueError(
                "Invalid unit (valid units are `seconds`, `minutes`, `hours`, "
                "`days`, and `weeks`)"
            )

        if self.latest is not None:
            if not (self.latest >= self.interval):
                raise ScheduleError("`latest` is greater than `interval`")
            interval = random.uniform(self.interval, self.latest)
            # interval = random.randint(self.interval, self.latest)
        else:
            interval = self.interval

        self.period = datetime.timedelta(**{self.unit: interval})
        self.next_run = datetime.datetime.now() + self.period
        if self.start_day is not None:
            if self.unit != "weeks":
                raise ScheduleValueError("`unit` should be 'weeks'")
            weekdays = (
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            )
            if self.start_day not in weekdays:
                raise ScheduleValueError(
                    "Invalid start day (valid start days are {})".format(weekdays)
                )
            weekday = weekdays.index(self.start_day)
            days_ahead = weekday - self.next_run.weekday()
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            self.next_run += datetime.timedelta(days_ahead) - self.period
        if self.at_time is not None:
            if self.unit not in ("days", "hours", "minutes") and self.start_day is None:
                raise ScheduleValueError("Invalid unit without specifying start day")
            kwargs = {"second": self.at_time.second, "microsecond": 0}
            if self.unit == "days" or self.start_day is not None:
                kwargs["hour"] = self.at_time.hour
            if self.unit in ["days", "hours"] or self.start_day is not None:
                kwargs["minute"] = self.at_time.minute
            self.next_run = self.next_run.replace(**kwargs)  # type: ignore
            # Make sure we run at the specified time *today* (or *this hour*)
            # as well. This accounts for when a job takes so long it finished
            # in the next period.
            if not self.last_run or (self.next_run - self.last_run) > self.period:
                now = datetime.datetime.now()
                if (
                    self.unit == "days"
                    and self.at_time > now.time()
                    and self.interval == 1
                ):
                    self.next_run = self.next_run - datetime.timedelta(days=1)
                elif self.unit == "hours" and (
                    self.at_time.minute > now.minute
                    or (
                        self.at_time.minute == now.minute
                        and self.at_time.second > now.second
                    )
                ):
                    self.next_run = self.next_run - datetime.timedelta(hours=1)
                elif self.unit == "minutes" and self.at_time.second > now.second:
                    self.next_run = self.next_run - datetime.timedelta(minutes=1)
        if self.start_day is not None and self.at_time is not None:
            # Let's see if we will still make that time we specified today
            if (self.next_run - datetime.datetime.now()).days >= 7:
                self.next_run -= self.period


class ThreadScheduler(Scheduler):
    def __init__(self, name=None, final_task=None):
        """
        Wrapper around the schedule module for use within the subsystem classes
        that comes with its own "run_pending" loop.

        Parameters
        ----------
        name: str, optional
            used as an identifier for the Daemon, no other function like this.
        final_task: function or None, optional
            A function which is executed once the loop in `_run_pending_loop` stops. If None (default), do nothing.
        """
        Scheduler.__init__(self)  # create the Scheduler instance
        self.name = type(self).__name__ if name is None else name
        self.logger = logging.getLogger(f'{type(self).__name__}{"" if name is None else "_" + name}')
        self.logger.debug('Initialised Class')

        # def variable for thread
        self.loop_thread = threading.Thread(target=self._run_pending_loop, name=f'{type(self).__name__}-{self.name}')
        self.exit_event = threading.Event()
        self.lock = threading.Lock()

        self.final_task = final_task

        self._trigger_run_all_delay_seconds = None

        self.thread_should_start = False  # a variable which signalize the Job if they should start the loop

    def __del__(self):
        try:
            self.logger.debug("__del__")
            self.clear()
            self.stop()
        except NameError:  # workaround as open() is deleted from the garbage collector already
            pass

    def exit(self):
        self.__del__()

    @property
    def is_active(self):
        return self.loop_thread.is_alive()

    @property
    def _exit_event_is_set(self):
        # The is_set() call in a lock which makes it possible to exit the exit_event.wait() without killing the loop
        with self.lock:
            return self.exit_event.is_set()

    def every(self, interval: float = 1., start=True):
        """
        Schedule a new periodic job - the ThreadScheduler version

        :param interval: A quantity of a certain time unit
        :param start: Defines it the loop thread starts, too. Default is True.
        :return: An un-configured :class:`Job <Job>`
        """
        # the Job registers itself to the scheduler `jobs`-list -> if set, the job has to start the loop after this step
        # -> self.thread_should_start signalizes this to the job,
        # Its basically something like `ThreadJob(interval, self, start)` but as overload should have the attributes.
        self.thread_should_start = start
        # noinspection PyTypeChecker
        job = ThreadJob(interval, self)
        return job

    def update_event_wait(self):
        # Exit the exit_event.wait() in the loop to update the waiting time. The lock is there to not stop the loop.
        with self.lock:
            self.exit_event.set()
            self.exit_event.clear()

    def clear(self, *args, **kwargs):
        self.logger.debug(f'Clear {self.name} scheduler.')
        Scheduler.clear(self, *args, **kwargs)
        if len(self.jobs) < 1:  # stop if no jobs are registered
            self.stop()  # stop the loop if it is running

    def run_all(self, delay_seconds: int = 0) -> None:
        """Run all jobs inside the thread. No conflict with other threads."""
        self._trigger_run_all_delay_seconds = delay_seconds
        self.update_event_wait()

    def _run_pending_loop(self):
        """
        Method is checking the schedule and is running pending tasks.
        """
        try:  # not clear why this 'try' is needed
            while not self._exit_event_is_set:  # the loop of this thread
                if self._trigger_run_all_delay_seconds is not None:
                    try:
                        Scheduler.run_all(self, delay_seconds=self._trigger_run_all_delay_seconds)
                    except (RuntimeError, Exception):  # when something went wrong
                        var = traceback.format_exc().strip().replace('\n', '/n ')
                        self.logger.warning(f'Schedule {self.name} run all jobs crashed with traceback: {var}')
                    finally:
                        self._trigger_run_all_delay_seconds = None  # reset the flag
                else:
                    try:
                        self.run_pending()  # start all pending jobs
                    except (RuntimeError, Exception):  # when something went wrong
                        var = traceback.format_exc().strip().replace('\n', '/n ')
                        self.logger.warning(f'Schedule {self.name} run pending job(s) crashed with traceback: {var}')
                        self.exit_event.wait(1)  # adds 1 if something failed, otherwise things can go crazy

                if self.next_run is None:  # older scheduler versions crash at 'idle_seconds' when nothing scheduled
                    self.logger.debug(f'Started loop without jobs.')
                    self.exit_event.wait()  # wait until update_event_wait
                    # if len(self.jobs) < 1:  # stop if no jobs are registered
                    #     self.stop()

                elif self.idle_seconds > 0:
                    self.exit_event.wait(self.idle_seconds)

        except (RuntimeError, Exception):  # when something went wrong
            var = traceback.format_exc().strip().replace('\n', '/n ')
            self.logger.warning(f'Schedule {self.name} loop crashed with traceback: {var}')

        else:  # Execute if while try ended without exception, i.e. loop stopped
            self.logger.debug(f'Schedule loop stopped; {self.name}')

        finally:
            if self.final_task is not None:
                self.final_task()

    def start(self):
        # start loop thread
        self.logger.info(f'Start scheduler; {self.name}')
        if not self.loop_thread.is_alive():
            self.exit_event.clear()  # self.exit_event = threading.Event()
            self.loop_thread = threading.Thread(target=self._run_pending_loop,
                                                name=f'{type(self).__name__}-{self.name}')
            self.loop_thread.start()

    def stop(self, join=True):
        # terminate loop thread
        self.logger.info(f'Stop scheduler; {self.name}')
        self.exit_event.set()  # stop the loop
        if self.loop_thread.is_alive():  # if thread is active
            # wait for thread to finish, but only if stop() is executed by another thread and not self.loop_thread
            # this could happen if you schedule an event to disable the scheduler. CAUTION, this can't prevent more
            # complex situations, i.e. schedule a job which stops a thread which again stops the scheduler.
            # Use join=False in such situations.
            if join and threading.currentThread().getName() != self.loop_thread.getName():
                self.loop_thread.join()

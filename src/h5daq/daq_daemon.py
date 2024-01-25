#!/usr/bin/python3

# Author: Kilian Holzapfel <kilian.holzapfel@tum.de>

import datetime
import logging
import os
import subprocess
import threading
import time
import traceback

import h5py
import numpy as np
from dateutil.relativedelta import relativedelta

from .argparser import DAQDaemonParser
from .scheduler import ThreadScheduler


# TODO: can this be deleted? looks like not used, but not sure at this point
# def shutdown(signum, frame):  # signum and frame are mandatory
#    sys.exit(0)


class DAQDaemon(DAQDaemonParser):
    # A h5daq global Lock (threading) for the hdf5 access. h5py don't seem to be 100% thread prove which cause the
    # access lock not to be reset, probably.
    h5py_lock = None

    def __init__(self, job_list,
                 name='',
                 write_period: int = 60,
                 write_period_jitter: int = None,
                 directory='.',
                 file_format_str="DAQ_{date_from}_{date_to}.hdf5",
                 rollover_interval_dict=None,
                 compression_dict=None,
                 attrs=None):
        """ DAQDaemon controls:
        1. repeated measurements (or jobs) at a given frequency. The different measurement (items of job_list) are
           threaded, so running parallel. The jobs can share a lock to prevent them running in parallel.
        3. collecting all job-buffers and writing them to a hdf5 file after a constant period in seconds (write_period)

        Parameters
        ----------
        job_list: list of DAQJob instances,
            each job is in charge of one measurement
        name: str, optional
            used as an identifier for the Daemon, no other function like this.
        directory: str, optional
            sets the data path where hdf5 files are saved. Default, current folder. Raise a KeyError if not existing.
        file_format_str: str, optional
            used to create the filename, has to included at least one or all placeholders: '{date_from}', '{date_to}'.
            Those placeholders are replaced with the UTC time in ms precision, i.e. 20200101T125959.001.
            '{date_from}': start time; '{date_to}': end time. Default: "DAQ_{date_from}_{date_to}.hdf5".
        write_period: Union[int, tuple, list], optional
            period between writing the buffer to the file in seconds. To define a period randomly taken from an interval
            for every iteration, set a write_period_jitter.
        write_period_jitter: int, optional
            This parameter adds a jitter to the write period. This means, every write interval is randomly taken from
            the interval `write_period+-write_period_jitter`. This can help to minimize collisions with other events
            that access the hdf5 file. If None (default) the write_period has no jitter.
        rollover_interval_dict: dict or None, optional
            defines the rollover interval of the filename. When the rollover happens the next rollover is calculated.
            Dict has to contain exactly one key from: ['years','months','weeks','days','hours','minutes','seconds'].
            Example: rollover_interval_dict={"days": 1} (default)
            1 days  -> rollover happens at 00:00:00 independent when DAQDaemon started (default)
            2 hours -> rollover happens at +2h from DAQDaemon start round down to **:00:00.
        compression_dict: dict, optional
            The dict which specifies the compression for all datasets of the hdf5 file.
            The dict can be 'None': NO compression (Default).
            Or it has to have this two entries, i.e. {'compression': 'gzip', 'compression_opts': 6}
        attrs : dict, optional
            attrs are attributes (similar ot datasets) saved in the group of the DAQJob of the hdf5 file.
            E.g. an ID of a specific device.
        """
        self.logger = logging.getLogger(f'{type(self).__name__}-{name}')
        self.logger.debug('Initialised Class')

        DAQDaemonParser.__init__(self, name)  # gives: self.name, handle_sdaq_cmdline, create_subsys_argparser
        self.scheduler = ThreadScheduler(f'{type(self).__name__}-{name}', final_task=self._final_write)
        if DAQDaemon.h5py_lock is None:
            DAQDaemon.h5py_lock = threading.Lock()

        # variables
        self.directory = directory
        self.file_format_str = file_format_str
        self.file_name = ""  # set automatically, never set it manually
        self.file_error_count = 0  # counts up if it fails to open the file, gets back to 0 if it succeeded

        if compression_dict is None:
            self.compression_dict = {'compression': None,
                                     'compression_opts': None}
        else:
            self.compression_dict = {'compression': compression_dict['compression'],
                                     'compression_opts': compression_dict['compression_opts']}
            if 'shuffle' in compression_dict:
                self.compression_dict.update({'shuffle': compression_dict['shuffle']})

        # write period parameters
        self._write_period = None
        self._write_period_jitter = None
        self.set_write_period(write_period=write_period, write_period_jitter=write_period_jitter)

        # rollover variables
        if rollover_interval_dict is None:
            rollover_interval_dict = {'days': 1}
        self.rollover_interval_dict = rollover_interval_dict
        self.time_file_rollover = None  # time when the next rollover of the filename happens

        # give each job the correct Locks, if two jobs have the same thread_id, they get the same Lock
        read_lock_dict = {}
        for job_i in job_list:  # over all thread_ids
            if job_i.thread_id is None:  # Job doesn't share a Lock with another Job
                pass  # the Job has its own unique Lock already
            elif job_i.thread_id not in read_lock_dict:  # if there is no RLock defined for thread_id, do it, else pass
                read_lock_dict.update({job_i.thread_id: threading.Lock()})
            else:  # the Lock for the thread_id exits
                job_i.read_lock = read_lock_dict[job_i.thread_id]

        self.job_list = job_list

        self.attrs = {}
        self.attrs_default = attrs  # attrs are attributes saved at the root group of the hdf5 file.

    @property
    def write_period(self):
        return self._write_period

    @property
    def write_period_jitter(self):
        return self._write_period_jitter

    def set_write_period(self, write_period: int, write_period_jitter: None):
        """Setter for write_period and write_period_jitter."""
        if not isinstance(write_period, (int, float)):
            raise TypeError(f'write_period must be one of type int or float; got type: {type(write_period)}')

        if isinstance(write_period_jitter, (int, float)):
            if write_period <= write_period_jitter:
                raise ValueError(f'write_period must be greater than write_period_jitter; got: {write_period}, '
                                 f'{write_period_jitter}; respectively')
        elif write_period_jitter is not None:
            raise TypeError(f'write_period_jitter must be one of type int, float or None; got type:'
                            f' {type(write_period)}')

        self._write_period = write_period
        self._write_period_jitter = write_period_jitter

    def _init_attrs_(self):
        self.attrs = {'write_period': self.write_period, 'write_period_jitter': self.write_period_jitter,
                      'name': self.name,
                      'rollover_interval': str(self.rollover_interval_dict),
                      'run_start': np.nan, 'run_end': np.nan,
                      'file_start': np.nan, 'file_end': np.nan,
                      'previous_file_id': np.nan, 'file_id': np.nan, 'following_file_id': np.nan}
        if self.attrs_default is not None:
            self.attrs.update(self.attrs_default)

    def _jobs_with_data(self, return_all=False):
        """Check and return the list of jobs which have data in the buffer internally."""
        jobs = []
        for job_i in self.job_list:
            if job_i.get_buffer_position(return_all=return_all, until_timestamp=self.time_file_rollover) > 0:
                jobs.append(jobs)

        return jobs

    # def _write_job_to_file(self, job, return_all=False):
    #     # TODO: under construction
    #     # get the job buffer and reset it internally
    #     buffer2write = job.get_buffer(return_all=return_all, until_timestamp=self.time_file_rollover)
    #
    #     write_size = 0
    #
    #     if buffer2write[0].shape[0] != 0:  # same as buffer2write[0] != []; buffer2write[0] is the 'time'
    #         # self.logger.debug(f'{job_i.group} - {current_time_file_rollover} - {data_arr[0, :]} - {data_arr}')
    #
    #         # get the group of the job or create it
    #         if job.group in self.file:
    #             group = self.file.get(job.group)
    #         else:
    #             # at the first time when the data of a job are added to the file
    #             group = self.file.create_group(job.group)
    #             if job.attrs is not None:
    #                 group.attrs.update(job.attrs)  # write attributes to group
    #
    #         for i, label_i in enumerate(job.label):
    #             # get table or create it, if it doesn't exist
    #             if label_i in group:
    #                 table = group[label_i]
    #             else:
    #                 # max of maxshape=2**64, see: https://docs.h5py.org/en/stable/high/dataset.html
    #                 # maxshape: 2**32 = ~4 days with 10kHz
    #                 # chunks: None give big files ~1GB (multiple datasets) -> define chunks=2**4
    #                 # libver='latest' create big files also when there is no data e.g. {maxshape=2**32, chunks=2**4}
    #                 # -> 91GB vs >1MB; this comes from maxshape, give it a None solves it.
    #                 table = self.file.create_dataset(job.group + '/' + label_i,
    #                                                  job.shape_list[i],
    #                                                  dtype=job.dtype_list[i],
    #                                                  maxshape=(None, *job.shape_list[i][1:]),  # see docs above
    #                                                  chunks=job.chunk_length_list[i],
    #                                                  fletcher32=True,  # enable checksum
    #                                                  **self.compression_dict)
    #
    #             table.resize(table.shape[0] + buffer2write[0].shape[0], axis=0)  # resize
    #             table[-buffer2write[0].shape[0]:] = buffer2write[i]  # append the new data
    #
    #             # to log the written data size in [bytes]
    #             write_size += buffer2write[i].nbytes
    #
    #     return buffer2write[0].shape, write_size

    # def _open_file(self):
    #     if self.file is None:
    #         try:
    #             self.file = h5py.File(self.file_name, 'a')
    #
    #         except (OSError, Exception):
    #             self.file_error_count += 1  # count one up
    #             if self.file_error_count < 5:
    #                 level = logging.DEBUG
    #             elif self.file_error_count < 10:
    #                 level = logging.WARNING
    #             else:
    #                 level = logging.CRITICAL
    #                 # TODO: doesn't solve the problem
    #                 # self._reset_h5_lock(self.file_name)  # reset the hdf5 flag
    #
    #             var = traceback.format_exc().strip().replace('\n', '/n ')
    #             self.logger.log(level, f'Failed to open file; count={self.file_error_count}; traceback: {var}')
    #
    #             return
    #
    #     else:
    #         self.file_error_count = 0

    # def _write_hdf5_2(self, return_all=False):
    #     # TODO: under construction
    #     logger_list = []
    #     write_size = 0
    #     jobs = self._jobs_with_data(return_all=return_all)
    #
    #     if not jobs:  # same as `jobs == []`
    #
    #         return
    #
    #     if self.file is None:
    #         self._open_file()
    #
    #     for job_i in jobs:
    #         shape, write_size_i = self._write_job_to_file(job=job_i, return_all=return_all)
    #         logger_list.append(f"{job_i.group} len: {shape}")
    #         write_size += write_size_i

    def _write_hdf5_lock(self, *args, **kwargs):
        """This use a h5daq global lock of accessing hdf5 files as h5py isn't thread save."""
        with self.h5py_lock:
            self._write_hdf5(*args, **kwargs)

    def _write_hdf5(self, return_all=False, flush_attrs=False):
        """collect the data from all Jobs, check for file name rollover, handles writes to hdf5 file
        PARAMETER
        ---------
        return_all: bool, optional
            SDAQ_job(s) get_buffer(return_all) to control if the buffer returns all entries or keeps some for
            compression. Default False.
        flush_attrs: bool, optional
            If set, it writes the hdf5 attributes to the file. This shouldn't happen to often, as it generates big hdf5
            files. This parameter is internally only used in _final_write() when the Daemon stops the file.
            Default False.
        """
        logger_dict = {"write_size": 0, "job_written_list": [],
                       "write_time": time.time()}  # save here the start time, subtract the end time later
        old_file_name = None  # save as self._update_file_name() overrides and remove it later in case its empty

        # try to open file or files
        # len(file)=1 no rollover, len(file)=2 its an rollover with [old_file,new_file]
        new_file = not os.path.exists(self.file_name)  # flag if attrs are written, only when the file is created
        files = []
        try:
            # libver='latest' cause big files -> ('earliest', 'v110') the earliest version after 1.10.0
            files.append(h5py.File(self.file_name, 'a', libver='latest'))

            current_time_file_rollover = self.time_file_rollover  # save as self._update_file_name() overrides

            # check if a rollover is happening, in case open and append file to 'files'
            # files are open before any data are erased from the single jobs -> if opening a file files fails, no data
            # is lost and another try happens after the write interval
            if self.time_file_rollover <= datetime.datetime.utcnow().timestamp():  # to guarantee datetime in utc
                old_file_name = self.file_name  # save as self._update_file_name() overrides

                self._update_file_name()  # this will update the self.time_file_rollover and self.file_name

                self.logger.info(f'Do rollover from {old_file_name} to {self.file_name}')

                # open new file as well
                files.append(h5py.File(self.file_name, 'a', libver='latest'))  # append new_file

                return_all = True  # to save all groups at least once per file when 'compress' is enabled

        # if it fails to open file(s), count, and log with increasing level
        except (OSError, Exception):
            self.file_error_count += 1  # count one up
            if self.file_error_count < 5:
                level = logging.DEBUG
            elif self.file_error_count < 10:
                level = logging.WARNING
            else:
                level = logging.CRITICAL
                self._reset_h5_lock(self.file_name)  # reset the hdf5 flag

            var = traceback.format_exc().strip().replace('\n', '/n ')
            self.logger.log(level, f'Failed to open file; count={self.file_error_count}; traceback: {var}')

        # was successfully to open file(s)
        else:
            for file_i in files:
                file_i.swmr_mode = True

            # Write attributes to file. It's important to do this just a few as possible.
            # Writing it at every write, the file size gets very big
            if len(files) > 1:  # when rollover is happening: write attrs to old and new file
                new_file_id = np.random.randint(2 ** 64 - 1, dtype=np.uint64)
                old_attrs = dict(self.attrs)  # copy the dict
                old_attrs.update({'file_end': current_time_file_rollover - .001,  # - 1ms
                                  'following_file_id': new_file_id})
                self.attrs.update({'file_start': current_time_file_rollover,
                                   'previous_file_id': self.attrs['file_id'],
                                   'file_id': new_file_id})
                files[0].attrs.update(old_attrs)
                files[1].attrs.update(self.attrs)

            # when the daemon started (new_file) or writes the last time (flush_attrs) to the file.
            elif flush_attrs or new_file:
                files[0].attrs.update(self.attrs)

            # reset error count
            self.file_error_count = 0

            # for each job get the buffer and write to the file(s)
            for job_i in self.job_list:
                # get the job buffer and reset it internally
                buffer2write = job_i.get_buffer(return_all=return_all)  # get buffered data and clear it

                if buffer2write:  # same as buffer2write[0] != []; buffer2write[0] is the 'time'
                    # self.logger.debug(f'{job_i.group} - {current_time_file_rollover} - {data_arr[0, :]} - {data_arr}')
                    mask_data_rollover = buffer2write[0] < current_time_file_rollover  # mask: which data to which file

                    logger_dict["job_written_list"].append(f"{job_i.group} len: {buffer2write[0].shape}")

                    for file_i in files:
                        len_append_items = np.sum(mask_data_rollover)
                        if len_append_items > 0:  # check if there is data_arr[:,mask_data_rollover]
                            # get the group of the job or create it
                            if job_i.group in file_i:
                                group = file_i.get(job_i.group)
                            else:
                                group = file_i.create_group(job_i.group)
                                if job_i.attrs is not None:
                                    group.attrs.update(job_i.attrs)  # write attributes to group

                            for i, label_i in enumerate(job_i.label):
                                # get table or create it, if it doesn't exist
                                if label_i in group:
                                    table = group[label_i]
                                else:
                                    # max of maxshape=2**64, see: https://docs.h5py.org/en/stable/high/dataset.html
                                    # maxshape: 2**32 = ~4 days with 10kHz
                                    # chunks: None give big files ~1GB (multiple datasets) -> define chunks=2**4
                                    # libver='latest' create big files also when there is no data
                                    # e.g. maxshape=2**32, chunks=2**4 -> 91GB vs >1MB
                                    if job_i.dtype_list[i] == str:
                                        dtype = h5py.special_dtype(vlen=str)
                                    else:
                                        dtype = job_i.dtype_list[i]

                                    table = file_i.create_dataset(job_i.group + '/' + label_i,
                                                                  job_i.shape_list[i],
                                                                  dtype=dtype,
                                                                  maxshape=(None, *job_i.shape_list[i][1:]),  #
                                                                  chunks=job_i.chunk_length_list[i],
                                                                  fletcher32=True,  # enable checksum
                                                                  **self.compression_dict)

                                table.resize(table.shape[0] + len_append_items, axis=0)  # resize
                                error_str = ''
                                for write_loop in range(3):
                                    try:
                                        # append the new data
                                        table[-len_append_items:] = buffer2write[i][mask_data_rollover]
                                        break
                                    except OSError:
                                        error_str = f'dset: {label_i}; pos: {table.shape[0]}; ' \
                                                    f'len_new: {len_append_items};'
                                        var = traceback.format_exc().replace('\n', '/n ')
                                        self.logger.warning(
                                            f'Writing data ({write_loop}) "{job_i.group}/{label_i}" failed with: {var}')
                                        pass

                                # add error information to attrs
                                if error_str != '':
                                    if 'write_error' in self.attrs:
                                        self.attrs['write_error'] += error_str
                                    else:
                                        self.attrs['write_error'] = error_str

                                # to log the written data size in [bytes]
                                logger_dict["write_size"] += buffer2write[i].nbytes

                        mask_data_rollover = ~mask_data_rollover  # invert mask if the rollover happened

        # in any case (also if Error is in 'else:') it closes both files
        finally:
            # self.logger.debug(f'Close files: {[i.filename for i in files]}')
            for files_i in files:
                files_i.flush()
                files_i.close()  # close the files

            # remove if the 'old' file if it is empty and a rollover happened;
            if old_file_name is not None:
                self._remove_emtpy_file(old_file_name)

        if logger_dict['write_size'] > 0:
            logger_dict["write_time"] = time.time() - logger_dict["write_time"]
            logger_dict.update({'name': self.name, 'filename': self.file_name})
            out_str = 'write to hdf5: name={name}, write={write_time:.3f}s,' \
                      'size={write_size:d}, groups={job_written_list:}, file={filename}'
            self.logger.debug(out_str.format(**logger_dict))
        else:
            self.logger.debug(f'Write to hdf5, no data, group(s) {[job_i.group for job_i in self.job_list]}')

    def _final_write(self):
        """ For the final write, first stop all jobs, collect the data and write it to the file the last time."""
        for job_i in self.job_list:
            job_i.stop()  # only constant_read jobs need this, however doing it for all jobs is easier

        self._write_hdf5_lock(return_all=True)  # last write to hdf5

    def _reset_h5_lock(self, file_name):
        """This function use the hdf5 package and tries to clear the write-lock-flag. So far this doesn't help."""
        proc = subprocess.run(f'h5clear -s {file_name}', shell=True, capture_output=True)  # reset the file lock
        rc, stdout, stderr = proc.returncode, '', ''
        if proc.stdout is not None:
            stdout = proc.stdout.decode("utf-8").strip()  # STDOUT to String
        if proc.stdout is not None:
            stderr = proc.stderr.decode("utf-8").strip()  # STDOUT to String
        self.logger.debug(f'Resetting the file lock; rc:{rc}, stdout: {stdout}; stderr: {stderr}')

    def _remove_emtpy_file(self, file_name, size=800):
        """Removes an empty file, created by the daemon but no data have been written."""
        if os.path.getsize(file_name) <= size:
            try:
                file_size = os.path.getsize(file_name)
                os.remove(file_name)

            except Exception as a:
                self.logger.warning(f'Removing the empty SDAQ-File: {file_name} failed with: {a}')
            else:
                self.logger.info(f'Remove the empty SDAQ-File: {file_name} with Size: {file_size}')
                return True

        return False  # else return False

    def status(self):
        return self.scheduler.is_active

    @property
    def is_active(self):
        return self.scheduler.is_active

    def start(self, attrs=None):
        """Start the daemon. If it is running, it will pass. Attrs are saved at the root attributes of the hdf5 file.
        PARAMETER
        ---------
        attrs: None or dict, optional
            Attributes which are added to the root attributes of the hdf5 file. The attributes are deleted after stop().
            This is different to the attributes set at the daemon initialisation, which are added to all runs
            (run: start(), ..., stops()).
            """
        if not self.scheduler.is_active:
            self._update_file_name()  # set up file_name and rollover

            time_0 = datetime.datetime.utcnow().timestamp()  # to guarantee datetime in utc
            self._init_attrs_()
            if attrs is not None:
                self.attrs.update(attrs)
            self.attrs.update({'run_start': time_0, 'file_start': time_0,
                               'file_id': np.random.randint(2 ** 64 - 1, dtype=np.uint64)})

            # Start Jobs
            for job_i in self.job_list:
                # register the job in the scheduler or starts the loop if read_mode='constant'
                job_i.start()

            self.scheduler.clear()
            if self.write_period_jitter is None:
                self.scheduler.every(self.write_period).seconds.do(self._write_hdf5_lock)
            else:
                self.scheduler.every(
                    self.write_period - self.write_period_jitter).to(
                    self.write_period + self.write_period_jitter).seconds.do(self._write_hdf5_lock)

            return f'Starting the logger daemon in file: {self.file_name}'
        else:
            return 'Logger daemon is already running!'

    def stop(self):
        self.logger.info(f'Stop DAQDaemon {self.name}, active: {self.scheduler.is_active}')
        if self.scheduler.is_active:
            # give the file the right ending time
            # initially the file is named form start to calculated rollover time, the latter has to be replaced
            # when DAQ is stopped. DON't give tz=datetime.timezone.utc as self.time_file_rollover is UTC
            datetime_rollover = datetime.datetime.fromtimestamp(self.time_file_rollover)
            datetime_rollover -= datetime.timedelta(milliseconds=1)
            time_file_rollover_str = datetime_rollover.strftime('%Y%m%dT%H%M%S.%f')[:-3] + 'Z'

            # stop time
            datetime_stop = datetime.datetime.utcnow()  # to guarantee datetime in utc
            timestamp_stop = datetime_stop.timestamp()
            time_file_stop = datetime_stop.strftime('%Y%m%dT%H%M%S.%f')[:-3] + 'Z'
            new_file_name = self.file_name.replace(time_file_rollover_str, time_file_stop)

            print(time_file_rollover_str, time_file_stop, self.file_name, new_file_name)
            # self.logger.debug('start stopping sdaq')
            self.attrs.update({'run_end': timestamp_stop, 'file_end': timestamp_stop})
            self.scheduler.stop()

            # reset the time_file_rollover, must happen after self.scheduler.stop() -> final_write() ...
            self.time_file_rollover = None  # stop sdaq, so no time for next file rollover

            # in case, rename file
            # os.path.exists(self.file_name) as an empty file is deleted right away at creation
            if self._remove_emtpy_file(self.file_name):  # returns True if file was deleted, else False
                return f'Stopping the logger daemon - empty file removed'
            elif self.file_name != new_file_name and os.path.exists(self.file_name):
                self.logger.info(f'rename file from {self.file_name} to {new_file_name}')
                os.rename(self.file_name, new_file_name)
                old_file_name = self.file_name
                self.file_name = new_file_name  # update file_name, to be able to access it from the class

                return f'Stopping the logger daemon - file renamed from {old_file_name} to {new_file_name}'
            else:
                return f'Stopping the logger daemon - file {self.file_name}'
        else:
            return 'Logger daemon is not running!'

    def flush(self):
        """Collect the data from the jobs and write it to the hdf5 file. If Daemon is not active,
        it writes one file with {'date_from': date_now, 'date_to': date_now}"""
        if self.scheduler.is_active:
            self.scheduler.run_all()  # run the
        else:
            self._update_file_name()  # update time rollover, needed for _write_hdf5
            # mod file name
            datetime_now = datetime.datetime.now(tz=datetime.timezone.utc)
            date_now_str = datetime_now.strftime('%Y%m%dT%H%M%S.%f')[:-3] + 'Z'
            file_name_dict = {'date_from': date_now_str, 'date_to': date_now_str}
            self.file_name = os.path.join(self.directory, self.file_format_str.format(**file_name_dict))

            self._write_hdf5_lock(return_all=True)

    def _update_file_name(self, ):
        """update file name, calculate next rollover and in case (if {date_to} specified in format_str) integrate the
        rollover in the file name"""
        # get and update datetime_start,
        # if the start of the daemon -> no rollover, use the actual time. otherwise, it's the rollover time + .001s
        if self.time_file_rollover is None:
            datetime_now = datetime.datetime.utcnow()  # to guarantee datetime in utc
        else:
            # NO tz=datetime.timezone.utc as self.time_file_rollover is UTC
            datetime_now = datetime.datetime.fromtimestamp(self.time_file_rollover)

        datetime_rollover = self._cal_time_file_rollover(datetime_now)  # calculate the rollover time

        # save rollover time as timestamp to check if it's time to roll,
        # time.time() is faster as datetime.datetime.utcnow()
        self.time_file_rollover = datetime_rollover.timestamp()

        # convert time to ONC format, date_to_str is the millisecond before the rollover
        date_from_str = datetime_now.strftime('%Y%m%dT%H%M%S.%f')[:-3] + 'Z'
        date_to_str = (datetime_rollover - datetime.timedelta(milliseconds=1)).strftime('%Y%m%dT%H%M%S.%f')[:-3] + 'Z'

        file_name_dict = {'date_from': date_from_str, 'date_to': date_to_str}

        # string format with dict, i.e. file_name_dict['directory'] is entered at '{directory}'
        self.file_name = os.path.join(self.directory, self.file_format_str.format(**file_name_dict))

    @property
    def rollover_interval_dict(self):
        return self._rollover_interval_dict

    @rollover_interval_dict.setter
    def rollover_interval_dict(self, rollover_interval_dict):
        """ use to calculate the time of the filename rollover. If updated while the DAQDaemon is running,
        change of interval will happen after next rollover
        rollover_interval_dict : dict of int
            defines the interval, only one key is allowed, with value>=1
            'years','months','weeks','days','hours','minutes','seconds'"""

        valid_keys = ['years', 'months', 'weeks', 'days', 'hours', 'minutes', 'seconds']

        if len(rollover_interval_dict) != 1:  # check that there is only one entry in the dict
            raise KeyError(f'interval_dict must have exactly one key: received {rollover_interval_dict}')

        key = list(rollover_interval_dict.keys())[0]
        if key not in valid_keys:  # check if key is valid
            raise KeyError(f'round_interval_dict key {rollover_interval_dict.keys()} not from {valid_keys}')
        # value hast to be int, check it here or convert
        rollover_interval_dict[key] = int(rollover_interval_dict[key])
        if rollover_interval_dict[key] < 1:  # check that value is >1
            raise KeyError(f'round_interval_dict value has to be >1, given {rollover_interval_dict[key]}')

        self._rollover_interval_dict = rollover_interval_dict

    def _cal_time_file_rollover(self, datetime_start):
        """Returns the end of the interval the datetime_start is inside.
        The method is like datetime_start is rounded up and modulo to the given interval.
        e.g.:
        cal_time_file_rollover(datetime_start=XXXX-02-28 12:43:65.281932, round_interval_dict={'days': 1}) returns:
        datetime(XXXX-01-30 23:59:59.999000).timestamp()
        or in some cases, if (XXXX%4 and not XXXX%100) or XXXX%400 datetime(XXXX-02-29 23:59:59.999000).timestamp()

        PARAMETER
        ---------
        datetime_start : datetime

        RETURNS
        -------
        end_of_interval : float,
            Is the rounded up and modulo to the given interval of datetime_start.
            Returns the time in seconds since the Epoch as float, including milliseconds.
        """

        # don't change the order or delete any item of interval_dict_round!
        interval_dict_round = {'years': 0,
                               'months': -datetime_start.month + 1,  # target it January, month=1 and not month=0 -> +1
                               'weeks': 0,
                               'days': -datetime_start.day + 1,  # stars at 1 not 0, same as month, compensate here
                               'hours': -datetime_start.hour,
                               'minutes': -datetime_start.minute,
                               'seconds': -datetime_start.second,
                               'microseconds': -datetime_start.microsecond}

        # interval_dict_round hast to add the interval and round down (subtract the time).
        # For e.g. self.rollover_interval_dict = {'days':3}
        # -> interval_dict_round: 'years','month','weeks' have to be 0
        # -> interval_dict_round: 'days'=+3
        # -> interval_dict_round: everything<'smaller' day is the actual time - 1000 microseconds
        for i in interval_dict_round.keys():
            if i in self.rollover_interval_dict:
                if i == 'weeks':
                    interval_dict_round['days'] = -datetime_start.weekday()  # roll over happens Sunday at 23:59:59.999
                interval_dict_round[i] = self.rollover_interval_dict[i]
                break
            else:
                interval_dict_round[i] = 0

        # Here we use the 'relativedelta' module instead of 'datetime.timedelta' as 'datetime.timedelta' does not
        # support 'months' and 'years' which can cause problems for leap years
        return datetime_start + relativedelta(**interval_dict_round)

    @property
    def file_format_str(self):
        return self._file_format_str

    @file_format_str.setter
    def file_format_str(self, file_format_str):
        """Defines the filename. If updated while the DAQDaemon is running, change will happen after next rollover.

        PARAMETER
        ---------
        file_format_str : str
            defines the format of the file name, has to included at least one or all placeholders: '{date_from}',
            '{date_to}'. Those placeholders are replaced with the UTC time with ms precision.
            '{date_from}': start time; '{date_to}': end time."""

        valid_keys = ['{date_from}', '{date_to}']
        # check that if there is at least one of the valid keys set
        if not any(key_i in file_format_str for key_i in valid_keys):
            raise KeyError(f"file_format_str has to include at least one placeholder of {valid_keys}, "
                           f"given: {file_format_str}")

        self._file_format_str = file_format_str

    @property
    def directory(self):
        return self._data_path

    @directory.setter
    def directory(self, data_path):
        """Defines the data path where the file(s) are saved. If updated while the DAQDaemon is running,
        change will happen after next rollover/

        PARAMETER
        ---------
        directory : str
            defines the directory"""
        if os.path.exists(data_path):
            self._data_path = os.path.abspath(data_path)
        else:
            os.makedirs(data_path, exist_ok=True)
            self.logger.warning(f"Directory does not exist. Create it now. Given: {data_path}")

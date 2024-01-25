#!/usr/bin/python3

# Author: Kilian Holzapfel <kilian.holzapfel@tum.de>
import datetime
import fcntl
import logging
import os
import signal
import subprocess
import threading
import time
import traceback
import warnings

import numpy as np


# TODO: can this be deleted? looks like not used, but not sure at this point
# def shutdown(signum, frame):  # signum and frame are mandatory
#    sys.exit(0)


class DAQJob:
    """ The class to buffer a group of datasets. One DAQDaemon which consists of one or several DAQJob's, collect
        the datasets of the Jobs and saves the datasets in a hdf5 file or if the datasets exists already it appends with
        new data. The ''group'' defines the hdf5-directory under which all datasets are saved with their ''label''s.
        The ''getter'' is a function - e.g. which does a measurement - and which return a list of items which are added
        to the dataset (len(list) has to match len(datasets)). Each item can be a single value, a list or a numpy array.
        For the later two, the shape has to be set for each item correspondingly to set up the datasets.
        This class keeps care of storing the getter execution time (np.float64), if no time is provided by the getter
        (see parameter: provide_time) and setting it up as a dataset.
        Parameters
        ----------
        group : str
            sets the group inside the hdf5 file where the measurement data is saved.
            The resulting structure are datasets with the path "/{group}/{label[i]}" inside the hdf5 file.
        label : str or list of str
            The labels which correspond to the items the getter function returns <-> len(label) == len(getter())!
            If provide_time=True -> len(label)=len(getter())-1, see below at provide_time. 'time' isn't allowed as label
            as it is reserved for the time dataset which is added by the SDAJob internally.
        getter : Union[function, iterable]
            function, The function which does the measurement and returns a list of items with
            len(label) == len(getter()). If it returns None, this measurement is skipped, e.g. if there is no data, the
            data is invalid, a sensor readout failed or for other reasons.
        read_period : Int , optional
            The period in seconds between two measurements. Therefore, the measurement frequency is 1./read_period.
            Only for read_mode='schedule' otherwise read_period is ignored.
        read_mode: Bool , optional
            Available modes are: 'schedule', 'constant', 'trigger'. A read of items over the getter can be triggered
            either by the 'schedule'r after every read_period, a 'constant' loop over the getter (as fas as possible) or
            'trigger'ed externally by executing start() for each read of the items.
        thread_id : Int , optional
            Jobs with the same thread_id will block each other while Jobs with different thread_id's will work in
            'parallel' (different threads, no multiprocessing). The thread_id works only inside one SDAQDaemon instance
            and not system-wide or among all SDAQDaemon instances. If thread_id is None (default) the Job gets an own
            thread.
        shape : Union[tuple, list[tuple]] tuple of ints or list of tuple of ints, optional
            Defines the shape of the datasets. The first item of each tuple has to be 0, as this is the axis along the
            items are appended. If no shape is specified, all items are set to a 1D dataset, i.e. ``(0,)``(default).
            If one tuple is given e.g. ``(0, 3)``, the shape is used for all datasets beside 'time'.
            Otherwise, the shape has to be a list of tuples where each tuple defines the shape of the single datasets,
            e.g. [(0,2,3),(0,4)]. The shape of the time, i.e. '(0,)' is added automatically,
            therefore it must be excluded in the list.
        dtype : data-type or array of data-type, optional
            If no data-type is specified, all items are np.float32. ATTENTION: STRINGS are not supported.
        chunk_length : int, optional
            defines the length in the first dimension of the hdf5-chunk. The other dimensioned are copied from shape.
            Default is 2**6. This parameter can save space on disc. The length should grow with the total expected data
            amount of the Job in a rollover interval, and vice versa.
        constant_read_error_delay : float, optional
            Delay in seconds which applies if getter returned None or invalid data as getter may return too early.
            In other words, it sets the maximum readout frequency and prevents high CPU loads of many loops.
            Default = .1 [sec]
        error_count_limit : int, optional
            after witch detected error the 'handle_error_function' should be executed. Default is 2.
        handle_error_function : None or executable, optional
            the function executed after 'error_count_limit' reached the Default is None.
        compress : bool, optional
            If True (default), it doesn't save the returned data (measurement) from the getter(), if there is no change.
            It only stores the first and the last measurement for the time information but dumps the measurement
            in between. The comparison takes all data into account, which means, one change in the data is enough.
            I.e. the raw series of (d0=data0) d0,d0,d0,d0,d1,d1,d1,d2,d3,d3,d3 gets d0,d0,d1,d1,d2,d3,d3. If compress is
            False, it takes the raw series.
        provide_time : bool, optional
            If False (default) DAQJob saves the time when it executes the getter(). If True, the getter() returns
            the time as the first value of the list. In both cases, the 'time' isn't included in label as it is added by
            the SDAJob internally.
        initial_buffer_length : int, bool
            defines the initial buffer length. In case the size increases to fit the data, and it returns to the initial
            size when after get_buffer()
        extend_mode : bool, optional
            sets the mode how the buffer is growing. When `True` the buffer gets extended whereas `False` (Default)
            appends the buffer. With extended provide_time must be True (is set automatically) and compress is not
            supported (ignored). Append features both options but is more cpu heavier and therefore slower.
        attrs : dict, optional
            attrs are attributes (similar ot datasets) saved in the group of the DAQJob of the hdf5 file.
            E.g. an ID of a specific device.
            """

    def __init__(self, group, label, getter, read_period: float = 1, thread_id: int = None,
                 shape=(0,), dtype=np.float32, chunk_length: int = 2**6,
                 read_mode: str = 'schedule',
                 constant_read_error_delay: int = 1, error_count_limit: int = 2, handle_error_function=None,
                 compress: bool = True,
                 initial_buffer_length: int = 100, provide_time: bool = False, extend_mode: bool = False,
                 attrs=None):

        self.group = str(group)
        self.logger = logging.getLogger(type(self).__name__ + '-' + group)
        self.logger.debug('Initialised Class')

        self.getter = getter
        if not callable(getter):
            raise TypeError(f'Getter is not callable, got: {getter}')

        self.read_period = float(read_period)

        valid_read_mods = ['schedule', 'constant', 'trigger']
        if read_mode not in valid_read_mods:
            raise KeyError(f'read_mode={read_mode} not in {valid_read_mods}')
        self.read_mode = read_mode  # s=scheduled, c=constant_read, t=triggered

        # error handling
        self.constant_read_error_delay = constant_read_error_delay  # delay if on read got wrong, in constant mode
        self.error_count_limit = error_count_limit
        self.error_count = 0
        self.handle_error_function = handle_error_function

        # settings
        self.compress = bool(compress)
        self._compress_last_data = False  # flag if last data point haven't changed

        if thread_id is not None:
            self.thread_id = int(thread_id)
        else:
            self.thread_id = None

        self.provide_time = bool(provide_time)  # if getter provides time as first argument
        self.extend_mode = bool(extend_mode)

        if attrs is None:
            self.attrs = None
        else:
            self.attrs = dict(attrs)  # attrs are attributes saved in the group of the hdf5 file.

        # create variable for thread, set in start and run
        self.thread = threading.Thread(target=self._read_values,
                                       name=f'{type(self).__name__}-{self.group}'
                                       )
        self.exit_event = threading.Event()
        self.exit_event.set()  # set it, which means disabling the Job

        # check dtype and label, length has to match
        if not isinstance(label, (list, np.ndarray)):
            label = [label]
        number_datasets = len(label)

        if not isinstance(dtype, (list, np.ndarray)):
            dtype = [dtype] * number_datasets

        if len(dtype) != number_datasets:
            raise ValueError(f'len(dtype_list)={len(dtype)} len(label)={number_datasets}, must have same length')

        for i, dtype_i in enumerate(dtype):  # check types
            if not type(dtype_i) == type:
                raise TypeError(f"'{dtype_i}' is no type from 'dtype': {dtype}")

        # check shape - goal: list[tuple[int,]], with first item of tuple=0 i.e. [(0,),(0,5),(0,5,5)]
        if isinstance(shape, tuple):  # given one tuple, applied to all datasets
            shape = [tuple([0, *shape[1:]])] * number_datasets
        elif isinstance(shape, list):  # given a list of tuples, append time tuple
            for i, shape_i in enumerate(shape):
                shape[i] = (0, *shape_i[1:])
        else:
            raise TypeError(f"Shape is no tuple or list of tuples, got: '{shape}'")

        if len(shape) != number_datasets:
            raise ValueError(f'len(shape)={len(shape)} len(label)={number_datasets}, must have same length. '
                             f'Shape: {shape}')

        if initial_buffer_length < 1:
            raise ValueError(f"initial_buffer_length, has to be >1 : '{initial_buffer_length}'")

        # chunk_length, defines the first length in the first dimension of the chunk.
        # The other dimensioned are copied from shape.
        if chunk_length < 1:
            raise ValueError(f"chunk_length, has to be > 0 : '{chunk_length}'")

        # add time as first item with np.float64
        label = ['time', *label]
        self.label = [str(i) for i in label]  # convert (and if not possible raise error) to string
        self.dtype_list = [np.float64, *dtype]
        self.shape_list = [(0,), *shape]
        self.initial_buffer_length = initial_buffer_length

        if chunk_length <= 2**4:
            chunk_length_time = (2**4,)
        else:
            chunk_length_time = (chunk_length,)
        self.chunk_length_list = [chunk_length_time, *[(chunk_length, *i[1:]) for i in shape]]  # the time is set

        self.buffer_position = 0
        self.buffer = []
        for dtype_i, shape_i in zip(self.dtype_list, self.shape_list):
            if dtype_i == str:
                dtype_i = object
            self.buffer.append(np.zeros((self.initial_buffer_length, *shape_i[1:]), dtype=dtype_i))

        # variable to set the lock
        self.read_lock = threading.Lock()  # overwritten by SDAQDaemon later if thread_id is not None
        self.buffer_lock = threading.Lock()  # lock when the buffer is accessed

    @property
    def extend_mode(self):
        return self._extend_mode

    @extend_mode.setter
    def extend_mode(self, value):
        self._extend_mode = value
        if self._extend_mode:
            # only works this way for compress and provide_time
            self.compress = False  # maybe implemented later
            self.provide_time = True  # probably there is no way to get the time when something added with extend
            # self.read_mode = 'constant'  # other modes are maybe implemented later
            # self.read_period = .1  # is used to define how often the STDOUT should be checked

    @property
    def is_active(self):
        """Returns if the Job is active. This means, either the thread is still alive or the event isn't set.
        During startup, the thread isn't directly started and at the shutdown,
        the exit_event is set before the thread is terminated."""
        return self.thread.is_alive() or not self.exit_event.is_set()

    def _handle_error_function(self):
        try:
            self.logger.warning(f'Handle error function; error_count: {self.error_count}')
            self.handle_error_function()
            self.logger.debug(f'Handle error function done.')
        except (RuntimeError, Exception):
            var = traceback.format_exc().replace('\n', '/n ')
            self.logger.error(f'handle_error_function failed with: {var}')

    def _read_values(self):
        """executes 'getter()' (reads values) and stores those values in 'buffer'."""
        try:
            data = self._get_data()  # getter data
        except (RuntimeError, Exception):
            self.error_count += 1
            var = traceback.format_exc().replace('\n', '/n ')
            self.logger.warning(f'Getter failed with (error counter: {self.error_count}): {var}')

            if self.handle_error_function is not None and self.error_count_limit <= self.error_count:
                self._handle_error_function()

            # here to delay as getter may return to early
            if self.read_mode == 'constant':
                self.exit_event.wait(self.constant_read_error_delay)  # similar to time.sleep(), but can be terminated
                self._start_child_class()  # for DAQJobExtern to restart the process
            return

        # skipp measurement if data is None, else add it to the buffer
        if data is None:
            return

        try:
            if self.extend_mode:
                self._extend_buffer(data)
            else:
                self._append_buffer(data)
            self.error_count_append = 0
        except (RuntimeError, Exception):
            self.error_count += 1
            var = traceback.format_exc().replace('\n', '/n ')
            self.logger.warning(f'Append to buffer failed with (error counter: {self.error_count}): {var}')

            if self.handle_error_function is not None and self.error_count_limit <= self.error_count:
                self._handle_error_function()

            return

        self.error_count = 0

    def _get_data(self):
        # keep the lock as short as possible
        with self.read_lock:
            try:
                data = self.getter()

            # if getter fails, any Exception is a RuntimeError here
            except (RuntimeError, Exception):
                var = traceback.format_exc()
                raise RuntimeError(var)

        # if getter didn't fail, checks data, add it to the buffer or update only the time if self.compress
        if data is None or not data:  # skip this measurement, if 'None' or '[]'
            return None  # exit
        else:
            try:
                data = list(data)
            except TypeError:
                # interpret a single variable, e.g. int as a list -> getter() can return 1 and not [1]
                # but only if there is one variable expected, therefore time can't be provided by getter, too.
                raise TypeError(f'Getter does not return iterable object; got: {data}')

        # add time to data as first item
        if not self.provide_time:
            data = [datetime.datetime.utcnow().timestamp(), *data]  # to guarantee datetime in utc

        if not len(data) == len(self.label):
            raise ValueError(f'Len(output)={len(data)} must be len(label{"" if self.provide_time else "-1"})='
                             f'{len(self.label) if self.provide_time else len(self.label) - 1} : '
                             f'output={data}, label={self.label}, provide_time={self.provide_time}')

        return data

    def _extend_buffer(self, data):
        # now data has the right length including time as the first item
        # add this new data items to the buffer
        data_length = len(data[0])
        with self.buffer_lock:  # keep the lock as shot as possible
            # append the buffer when it is full to be able to fill more data in the next round
            if self.buffer_position + data_length >= self.buffer[0].shape[0]:
                new_length = self.buffer[0].shape[0] * 2  # calc new buffer length

                if new_length < data_length:  # just to be on the safe side
                    new_length += data_length

                for i, data_i in enumerate(data):
                    # resize the buffer( in place, i.e. 'self.buffer[i].resize' doesn't work as threads are used)
                    self.buffer[i] = np.resize(self.buffer[i],
                                               new_shape=(new_length, *self.buffer[i].shape[1:]))

            for i, data_i in enumerate(data):
                try:
                    self.buffer[i][self.buffer_position:self.buffer_position + data_length] = data_i

                except (RuntimeError, Exception):
                    var = traceback.format_exc().replace('\n', '/n ')
                    # self.logger.warning(f'Add data: {data} to buffer failed with: {var}')
                    raise ValueError(f'Add data: {data} to buffer failed with: {var}')

            # if compress: check if the actual and the last item in the buffer are the same
            # TODO: compress doesn't work for extend so far

            self._compress_last_data = False  # update _compress_last_data for next read/measurement
            self.buffer_position += data_length  # move the pointer to the buffer for next read/measurement

    def _append_buffer(self, data):
        # now data has the right length including time as the first item
        # add this new data items to the buffer
        with self.buffer_lock:  # keep the lock as shot as possible
            # append the buffer when it is full to be able to fill more data in the next round
            if self.buffer_position >= self.buffer[0].shape[0]:
                new_length = self.buffer[0].shape[0] * 2  # calc new buffer length
                for i, data_i in enumerate(data):
                    # resize the buffer( in place, i.e. 'self.buffer[i].resize' doesn't work as threads are used)
                    self.buffer[i] = np.resize(self.buffer[i],
                                               new_shape=(new_length, *self.buffer[i].shape[1:]))

            for i, data_i in enumerate(data):
                try:
                    self.buffer[i][self.buffer_position] = data_i

                except (RuntimeError, Exception):
                    var = traceback.format_exc().replace('\n', '/n ')
                    # self.logger.warning(f'Add data: {data} to buffer failed with: {var}')
                    raise ValueError(f'Add data: {data} to buffer failed with: {var}')

            # if "compress"ed: check if the actual and the last item in the buffer are the same
            if self.compress and self.buffer_position != 0 and \
                    all([np.all(i[self.buffer_position - 1] == i[self.buffer_position]) for i in self.buffer[1:]]):

                # if flag is set update only the time, buffer_position states the same
                if self._compress_last_data:
                    # self.logger.debug('update only time')
                    self.buffer[0][self.buffer_position - 1] = self.buffer[0][self.buffer_position]

                # add another entry when it's the first time that the items at buffer_position-1 and buffer_position
                # only differ in time. So it there two items for the period when the parameters haven't changed.
                # One at with the start timing and the other with the end
                else:
                    self.buffer_position += 1  # move the pointer to the buffer for next read/measurement
                self._compress_last_data = True  # update _compress_last_data for next read/measurement

            else:
                self._compress_last_data = False  # update _compress_last_data for next read/measurement
                self.buffer_position += 1  # move the pointer to the buffer for next read/measurement

    def _thread_read(self):  # base class
        if not self.thread.is_alive():  # skip the read if thread is still alive
            self._thread_read_child_class()
            self.thread = threading.Thread(target=self._read_values,
                                           name=f'{type(self).__name__}-{self.group}-read')
            self.thread.start()

    def _thread_read_child_class(self, ):
        pass

    def _loop_read(self):  # base class
        if self.read_mode == 'constant':
            self._start_child_class()
            while not self.exit_event.is_set():  # the event is set to terminate the loop
                self._read_values()

        elif self.read_mode == 'schedule':
            while not self.exit_event.is_set():  # the event is set to terminate the loop
                t_0 = time.time()
                self._start_child_class()
                self._read_values()

                sleep_time = self.read_period - (time.time() - t_0)
                if sleep_time > 0:
                    self.exit_event.wait(sleep_time)  # similar to time.sleep(sleep_time), but can be terminated
                else:
                    self.exit_event.wait(.1)

        else:
            self.logger.warning('_loop_read should only be executed when read_mode: "constant" or "schedule"')
            pass

    def get_buffer(self, return_all=False, until_timestamp=None):  # base class
        """get data of buffer, and delete it, keep the lock as short as possible

        Return
        ----------
        buffer_return : list
            list of measurements [[time_0,*[values_0]],[time_1,*[values_1]],...]
        """
        # if active and compress, keep the last read for compression,
        buffer_return = []

        index_to_return = self.get_buffer_position(return_all=return_all, until_timestamp=until_timestamp)

        if index_to_return > 0:
            with self.buffer_lock:  # keep it as short as possible
                self.buffer_position -= index_to_return  # set the buffer position as items are removed

                for i, buffer_i in enumerate(self.buffer):  # loop over all buffer items
                    # split the buffer items for the return
                    buffer_return.append(buffer_i[:index_to_return])
                    # resize rest of buffer item to fit self.initial_buffer_length and shape.
                    self.buffer[i] = np.resize(self.buffer[i][index_to_return:], (self.initial_buffer_length,
                                                                                  *self.shape_list[i][1:]))

        return buffer_return

    def get_buffer_position(self, return_all=False, until_timestamp=None):
        """Returns the `available` buffer position. For compression the lattes item isn't returned by default, only if
        `return_all` is enabled. The returned value should be interpreted as np.array([...])[:value], which means
        0 means: empty buffer; 1: one item and so on.
        
        PARAMETER
        ---------
        return_all: bool, optional
            Defines if all items of the buffer should be taken into account (True) or only the `available`
            (False, default).
        until_timestamp: None or float, optional
            Defines if all items of the buffer should be taken into account (None, default) or only the entries
            with a timestamp equal or smaller as the given.
        RETURNS
        -------
        buffer_position: int
            Current buffer position. 0 means no `available` item in the buffer, 1 is one `available` item in the buffer 
            and so on.
          """
        with self.buffer_lock:  # keep it as short as possible
            if self.buffer_position == 0:
                buffer_position = 0
            elif not return_all and self.compress:
                buffer_position = self.buffer_position - 1  # keep last item in each array with correct shape and dtype
            else:
                buffer_position = self.buffer_position  # don't keep last item

            if until_timestamp is not None:
                # take the time (self.buffer[0]), mask with the time and count (sum) the True in the mask
                buffer_position = np.sum(self.buffer[0][:buffer_position] <= until_timestamp)

        return buffer_position

    def peek(self):  # base class
        """get data of buffer, but NOT delete it, keep the lock as short as possible

        Return
        ----------
        buffer_return : list
            list of measurements [[time_0,*[values_0]],[time_1,*[values_1]],...]
        """
        # if active and compress, keep the last read for compression,
        buffer_return = []

        with self.buffer_lock:  # keep it as short as possible
            for i, buffer_i in enumerate(self.buffer):  # loop over all buffer items
                # split the buffer items for the return
                buffer_return.append(buffer_i[:self.buffer_position])

        return buffer_return

    def trigger_read(self, ):
        """Triggers a read / execution of one getter. This is possible if read_mode is either 'schedule' or 'trigger'"""
        if not self.read_mode == 'constant':  # and not self.thread.is_alive():  # check if thread is_alive
            self._read_values()
        else:
            self.logger.warning(f"Job in constant mode, can't execute trigger read. read_mode:{self.read_mode}; "
                                f"thread.is_alive:{self.thread.is_alive()}")

    def start(self, ):
        """ if self.constant_read==True, the measurement happens as fast as possible
        else the measurement is scheduled with the provided scheduler from the SDAQDaemon.
        """
        self.logger.debug(f'Start SDAQ Job: {self.group} with read mode: {self.read_mode}')
        if not self.is_active:  # prevent from running 2 times
            self.exit_event.clear()  # this puts self.is_active to True

            if self.read_mode in ['constant', 'schedule']:
                self.thread = threading.Thread(target=self._loop_read, name=f'{type(self).__name__}-{self.group}-loop')
                self.thread.start()

            elif self.read_mode == 'trigger' and not self.thread.is_alive():
                self._thread_read()  # read values

        elif self.is_active:
            self.logger.warning("DAQJob is still active, can't be activated 2 times")
        else:
            self.logger.warning("Thread of Job is still active, thread has to finish before DAQJob can be activated")

    # placeholder for e.g. DAQJobExtern
    def _start_child_class(self):
        pass

    def stop(self):
        """Stops the Job and joins the thread if self.constant_read==True, else this function isn't doing anything"""
        self.logger.debug(f'Stop SDAQ Job: {self.group}')
        self.exit_event.set()  # this stops all waits and loops -> stops the thread -> self.is_active gets False
        self._stop_child_class()

        # wait for thread to finish, but only if stop() is executed by another thread and not self.thread
        # this could happen if you getter or error handler functions disable the job
        if self.thread.is_alive() and threading.currentThread().getName() != self.thread.getName():
            self.thread.join()  # wait for thread to finish

        self.logger.debug(f'Stop SDAQ Job: {self.group} done.')
        self._compress_last_data = False  # set flag to false, this will trigger another save when the Job starts again

    # placeholder for e.g. DAQJobExtern
    def _stop_child_class(self):
        pass


class DAQJobExtern(DAQJob):
    def __init__(self, group, label, getter: str, delimiter=' ', comment='#', **kwargs):
        """Similar to DAQJob with the getter is an external command/script. The script runs in its own process and
        Job thread catches the STDOUT of this script, checks the output and append it to the Job buffer.
        A line of STDOUT is interpreted as one measurement where 'line.split(delimiter)' must return the same number
        of values as defined in the labels. E.g.: labels=['A', 'B'] -> STDOUT: '1 2' -> ['1', '2']

        There are two modes depending on the design of the script and configured with 'constant_read':
        1) The script can operate in a constant mode: 'constant_read'=True, i.e. where the script is in endless loop
        returning values to the STDOUT. The script will be terminated by the Job if it should stop.
        2) The script can be operated in a non-constant mode: 'constant_read'=False, where the Job executes the script
         with the given frequency over the scheduler. Here the script can only return one value

        PARAMETER
        ---------
        group : str
            see DAQJob
        label : list of str
            see DAQJob
        getter : str
            a string with the script/command, the string can include arguments, e.g. './test.sh -i 10 -v'
        delimiter : str, optional
            a string which separates the values in a STDOUT line
        comment : str, optional
            a string which signalise a line in the STDOUT to be comment. The comment will be sent to the logger as
            Info ('logger.ingo(...)'). Due to the design this option is only available in 'constant_read'=True
        **kwargs : `DAQJob` properties, optional
        """
        DAQJob.__init__(self, group, label, getter=self.__dummy_getter__, **kwargs)

        # select getter
        self.command_list = getter.split()
        self.command_list[0] = os.path.abspath(self.command_list[0])
        if not os.path.exists(self.command_list[0]):
            self.logger.warning(f'file does not exist {self.command_list[0]}')

        # select extend or append
        if self.extend_mode:
            self.getter = self._getter_extend
        else:
            self.getter = self._getter_append

        self.logger = logging.getLogger(type(self).__name__ + '-' + self.group)
        self.logger.debug('Initialised Class')

        self.delimiter = str(delimiter)  # check if it is a string and convert it
        self.comment = str(comment)  # check if it is a string and convert it

        self.process = None
        self.stdout_buffer = b''  # buffer only used in extend mode

        self.sdaq_performance = {'loops': 0,
                                 'min_buffer_len': 100000,
                                 'max_buffer_len': 0,
                                 'sum_buffer_len': 0,
                                 'time_next_debug_print': time.time(),
                                 'delta_time_debug_print': 60,  # seconds
                                 }

        # set parameters for extend mode, i.e. np.genfromtext
        # the number of cols have to be specified, otherwise np.genfromtxt use the first line to get the number of cols
        # if the first line is corrupt, everything else fails
        if self.provide_time:
            range_i = range(0, len(self.shape_list))
        else:
            range_i = range(1, len(self.shape_list))
        self.entries_per_line = np.sum([np.prod([1, *self.shape_list[i][1:]]) for i in range_i])

        self.dtype_structured = []
        for i in range_i:
            if self.dtype_list[i] == str:
                dtype_i = 'S256'
            else:
                dtype_i = self.dtype_list[i]
            self.dtype_structured.append((self.label[i], dtype_i, (1, *self.shape_list[i][1:])))

        self.stdout_buffer = b''  # in extend, stores the beginning of the line when it wasn't reached in the read

    @property
    def extend_mode(self):
        return self._extend_mode

    @extend_mode.setter
    def extend_mode(self, value):
        self._extend_mode = value
        if self._extend_mode:
            # only works this way for compress and provide_time
            self.compress = False  # maybe implemented later
            self.provide_time = True  # probably there is no way to get the time when something added with extend
            self.read_mode = 'constant'  # other modes are maybe implemented later
            self.read_period = .1  # is used to define how often the STDOUT should be checked

    def _start_process(self):
        if self.process is not None and self.process.poll() is None:
            self.logger.warning(f'DAQJobExtern: {self.group} process still active, killing it; '
                                f'self.process: {self.process}; '
                                f'self.process.poll(): {self.process.poll()}')

            self.process.kill()
            self.process.wait()

        try:
            # set up process and run in background
            # 'preexec_fn=os.setsid',  # creates a process group, used for killing all
            # `preexec_fn = os.setsid` creates a process group, used for killing all, but it isn't thread safe, and
            # it should be replaced by start_new_session=True. https://docs.python.org/3/library/subprocess.html
            self.process = subprocess.Popen(self.command_list,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            # bufsize=int(2**14)  #  no improvement
                                            start_new_session=True,
                                            )
        except (FileNotFoundError, Exception):
            var = traceback.format_exc()
            self.logger.error(f'DAQJobExtern: {self.group} failed with {var}')
        else:
            if self.extend_mode:  # set up what the extend_mode needs
                self.stdout_buffer = b''  # reset the buffer (only used in extend mode)

                # set stdout to non-blocking otherwise stdout read's can block till the file is closed
                fd = self.process.stdout.fileno()
                fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    def __dummy_getter__(self):
        pass

    def _thread_read_child_class(self):
        self._start_process()

    def _start_child_class(self):
        # close the old streams
        if self.process is not None:
            self.process.stdout.close()
            self.process.stderr.close()
        self._start_process()
        self.sdaq_performance['time_next_debug_print'] = time.time() + self.sdaq_performance['delta_time_debug_print']

    def _stop_child_class(self):
        # terminate process, as 'self.process.terminate()' not always works with shell=True
        # 'self.process is None' = process was never initialised; 'self.process.poll() is None' = process is running
        if self.process is not None and self.process.poll() is None:
            # loop as long as it is needed to kill the process
            signal_sent = signal.SIGTERM
            timeout = 5
            while True:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal_sent)
                except ProcessLookupError:  # if the process ended between the 'if self.p...' and 'os.kill'
                    return
                else:
                    self.logger.debug(f'Process kill - signal sent {signal_sent} and waiting {timeout}s.')
                    try:
                        self.process.wait(timeout=timeout)  # wait for process terminated
                    except subprocess.TimeoutExpired:
                        pass
                    else:
                        self.logger.debug('Process kill - done.')
                        return

                    # increase frequency and signal level
                    timeout = 1
                    signal_sent = signal.SIGKILL

    def _save_performance(self):
        buffer_len = len(self.process.stdout.peek().split())
        self.sdaq_performance['loops'] += 1
        self.sdaq_performance['sum_buffer_len'] += buffer_len
        if self.sdaq_performance['min_buffer_len'] > buffer_len:
            self.sdaq_performance['min_buffer_len'] = buffer_len
        elif self.sdaq_performance['max_buffer_len'] < buffer_len:
            self.sdaq_performance['max_buffer_len'] = buffer_len

    def _print_performance(self, erase=True):
        if time.time() >= self.sdaq_performance['time_next_debug_print']:
            self.sdaq_performance['time_next_debug_print'] += self.sdaq_performance['delta_time_debug_print']
            average = float(self.sdaq_performance["sum_buffer_len"]) / self.sdaq_performance["loops"]
            self.logger.debug(f'DAQJobExtern: {self.group} performance: '
                              f'buffer_ave: {average:4.0f} ,'
                              f'buffer_min: {self.sdaq_performance["min_buffer_len"]:4d} ,'
                              f'buffer_max {self.sdaq_performance["max_buffer_len"]:4d}')

            # reset for next in case
            if erase:
                self.sdaq_performance.update({'loops': 0,
                                              'min_buffer_len': 100000,
                                              'max_buffer_len': 0,
                                              'sum_buffer_len': 0})

    def _getter_append(self):
        """executes 'getter()' (reads values) and stores those values in 'buffer'"""
        waiting_for_values = True
        output = b''  # process.stdout.readline() returns a b'' no '' aka String

        # loop until one line is detected (b'\n') and return it as a list if no comment.
        while waiting_for_values:
            output += self.process.stdout.readline()  # blocking till next line
            self._save_performance()

            self._print_performance()
            if output and output[-1] == 10:  # return only after detecting a b'...\n'[-1]=10, not before.
                # check if it is a comment line
                if output.startswith(bytes(self.comment, "utf-8")):
                    self.logger.info(f'DAQJobExtern: {self.group} comment line from script:'
                                     f' {output.decode("utf-8").strip()}')
                    output = b''  # this measurement is skipped in DAQJob
                else:
                    # so far, a valid measurement, return it (to DAQJob)
                    return self.__convert_buffer__([output])

            # pool is None if running else it returns the return_code, stdout.peek() reads what is there without waiting
            elif self.process.poll() is not None and self.process.stdout.peek() == b'':
                waiting_for_values = False

                # return_code=0 is normal exit; self.exit_event.is_set() means Job shouldn't stop. If so, log stderr.
                if not self.exit_event.is_set():
                    stdout_str = ''.join([i.decode() for i in self.process.stdout.readlines()])
                    error_str = ''.join([i.decode() for i in self.process.stderr.readlines()])
                    raise RuntimeError(f'DAQJobExtern: {self.group} stopped with rc: {self.process.poll()}; '
                                       f'err: {error_str}; stdout: {stdout_str}; output: {output}')

            elif not self.exit_event.is_set():  # self.is_active:
                self.logger.debug(f"DAQJobExtern: {self.group}, output:{output}")

    def _getter_extend(self):
        """The getter for the buffer extend mode"""
        # try to read the buffer non-blocking
        self.exit_event.wait(self.read_period)  # wait a bit for each loop, less CPU load.

        buffer_lines = self.process.stdout.readlines()

        # check if there is new data
        if buffer_lines:  # equals buffer_lines != []
            pass

        # elif: check if the process is still running
        # pool is None if running else it returns the return_code, return_code=0 is normal exit;
        # self.exit_event.is_set() means Job shouldn't stop. If so, log stderr.
        elif self.process.poll() is not None and not self.exit_event.is_set():
            error_str = ''.join([i.decode() for i in self.process.stderr.readlines()])
            raise RuntimeError(f'DAQJobExtern: {self.group} stopped with: {error_str}')

        # else: there is no new data. Equals to 'elif not buffer_lines:' <-> 'buffer_lines == []'
        else:
            return None

        # add beginning of line which was read in the last getter call
        if self.stdout_buffer != b'':
            buffer_lines[0] = self.stdout_buffer + buffer_lines[0]
            self.stdout_buffer = b''

        # save end of non-complete line which was read for the next getter call
        if buffer_lines[-1][-1] != 10:  # which is b'\n' but 10 = b'#1 2 3 5\n'[-1]
            self.stdout_buffer = buffer_lines.pop(-1)
        else:
            self.stdout_buffer = b''  # reset the buffer

        return self.__convert_buffer__(buffer_lines)

    def __convert_buffer__(self, buffer_lines):
        if buffer_lines:  # same as buffer_lines != []
            for i, line in enumerate(buffer_lines):  # check if there is a comment
                if line[0] == 35:  # which is b'#' but 35 = b'#1 2 3 5\n'[0]
                    self.logger.info(f'{line}')

                if line[-1] != 10:  # sometimes lines get split into 2 lines. Check if there is a '\n' = 10 at the end
                    buffer_lines[i] += buffer_lines.pop(i + 1)  # combine it with the next line

            try:
                # numpy gives a warning in np.genfromtxt if something
                with warnings.catch_warnings(record=True) as w:
                    # 'unpack' should be True, but see seems not to work for older numpy versions -> set it to False
                    data = np.genfromtxt(buffer_lines,
                                         invalid_raise=False,
                                         usecols=np.arange(self.entries_per_line, dtype=int),
                                         dtype=self.dtype_structured,
                                         unpack=False)

                    # format the warning and try to extract the line which caused the warning
                    out_str_list = []
                    for i in w:
                        out_str_list.append(f'{i.message}; len(buffer_lines)={len(buffer_lines)}')
                        # try to get the lines which caused the warning
                        for j in str(i.message).split('Line #')[1:]:
                            try:
                                line_number = int(j.split(' ', 1)[0])
                                out_str_list.append(f'buffer_lines[{line_number}]: {buffer_lines[line_number - 1]}')
                            except (RuntimeError, Exception):
                                pass
                    if w:
                        self.logger.info(f'{len(w)} warnings in getter_extend: {";".join(out_str_list)}')

                # workaround for `unpack=True` not works with structured arrays in older numpy versions
                # mimic unpack behaviour: `When used with a structured data-type, arrays are returned for each field.`
                # from https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html
                data = [data[dtype_i[0]].reshape((-1, *dtype_i[2][1:])) for dtype_i in self.dtype_structured]
                return data

            except (RuntimeError, Exception):
                var = traceback.format_exc().replace('\n', '/n ')
                self.logger.warning(f'Convert Buffer failed with: {var},  buffer_lines:{buffer_lines}')
                return None

        return None

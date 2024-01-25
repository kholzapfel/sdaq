## SDAQ

SDAQ is a standalone module which provides a DAQ system based on hdf5-files powered by ``h5py, numpy, logging, scheduler``.
The package consists of two main classes [DAQDaemon](#daqdaemon) and [DAQJob](#daqjob).

#### Table of Contents
1. [Example](#example)
2. [Installation](#installation)
3. [Code Structure](#code-structure)
4. [ToDo-List](#todo-list)

## Example

```python
import sdaq

# def getter function, has to return a list
def getter():
    return [1,2,3]

# set up the DAQJob
job = sdaq.DAQJob(group='job_1', label=['table_0', 'table_1', 'table_2'],getter=getter, read_period=5)

# set up the DAQDaemon
daq_daemon = sdaq.DAQDaemon(job_list=[job])

# start the Daemon, wait, and stop it 
daq_daemon.start()
# wait some time, e.g. time.sleep(3)
daq_daemon.stop()
```
For more examples have a lock in the [example folder](./examples).

## Installation

### Prepare Python package
This step shouldn't be necessary.
Go to the directory where [this README you are reading](README.md) is placed (more precise the [pyproject.toml](pyproject.toml), but this should be the same) and run:
```bash
python3 -m build
```
This will create the files located in the folder `.egg-info`.

### Install with pip
Run the following command to install the package, and replace `.` with the root directory of the package. 
```bash
pip3 install -U --user -e .
```

#### AssertionError: Egg-link
In case you see an `AssertionError: Egg-link`, run:
```bash
rm ~/.local/lib/pythonX.X/site-packages/H5DAQ.egg-link
```
where X.X is your python version, e.g. 3.7 and run the command from **Install with pip** again.

This problem happens if you move the source code directory after the egg-link file is created.


## Code Structure

### DAQJob
The DAQJob is responsible for taking and buffering data. A single DAQJob can hold different datasets with different shapes. 
Internally the datasets are saved as numpy arrays. 

To add a new item to each dataset, a getter function is executed. This getter function has to be provided at the initialisation of the DAQJob ,and it should be a python function.
This getter function returns one entry for each dataset with the correct shape ,and single entries must be interpretable with the given dtype per data-set. 
Therefore, the length of the datasets along the first axis (axis=0) is the same for all datasets. The time when the getter is executed is saved automatically. It is also possible to provide the time from the getter as the first item.

For more information see the doc-string in [DAQJob (src file)](./src/sdaq/daq_job.py).

### DAQDaemon
The [DAQDaemon (src file)](./src/sdaq/daq_daemon.py), collects the buffered data from the [DAQJob(s)](#daqjob) and writes it to a hdf5 file, where each DAQJob gets its own hdf5-group (internal directory).
The DAQDaemon also runs the scheduler-loop for all scheduled DAQJob(s).

---
## TODO List:
* [ ] include options for logger with parameters (which file, rollover, level, fmt)
* [ ] 

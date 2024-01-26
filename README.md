# Simple Data Acquisition (SDAQ)

SDAQ (Simple Data Acquisition) is a Python package which provides a DAQ system based on hdf5-files powered by ``h5py, numpy, logging, scheduler``.
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
job = sdaq.DAQJob(
    group='job_1',  # the jobs name and also the name of the hdf5 group
    label=['table_0', 'table_1', 'table_2'],  # the table names and also the name of the hdf5 datasets
    getter=getter,
    read_period=5,  # in seconds, executes the `getter` every n seconds
)

# set up the DAQDaemon
daq_daemon = sdaq.DAQDaemon(job_list=[job])

# start the Daemon, wait, and stop it 
daq_daemon.start()
# wait some time, e.g. time.sleep(3)
daq_daemon.stop()
```
For more examples have a lock in the [example folder](./examples).

# Installation

## Installation directly from the repository
Pip can directly install the package from the repository. As pip only compares the version number and not the code, uninstall an existing installation before you install it from the repository. For updating the package just rerun the same commands. You can also specify the branch by changing master accordingly. The commands are:
```bash
pip3 install -U git+https://github.com/kholzapfel/sdaq.git@main
```
and to update it later on
```bash
pip3 install --no-deps --ignore-installed git+https://github.com/kholzapfel/sdaq.git@main
```

## Installation for developers
This installation downloads the source code, and the package loads directly from the source code for every import. Therefore, any changes to the code will have direct effect after an import.

Go to the directory of this README you are reading is placed (basically, to the directory of the pyproject.toml file, but this should be the same). Depending on your Python installation adopt python3/pip3 to python/pip, however python3 is required. Run:
```bash
mkdir /path/to/repros  # adopted the path, be aware that git clone creates a directory with the repository name
cd /path/to/repros # enter the directory

# clone/download the repository
git clone https://github.com/kholzapfel/sdaq.git@main  # downloads the repository
cd strawb_package  # enter the repository directory

# install the package
python3 -m build  # This will create the files located in the folder `.egg-info`
pip3 install -U -r requirements.txt  # install the required python packages
pip3 install -U --user -e .  # install the package in developer mode.
```

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
* [x] added basic example
* [ ] include options for logger
  * [x] which file
  * [x] level
  * [x] fmt
  * [ ] rollover

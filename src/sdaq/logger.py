# Author:  Kilian Holzapfel <kilian.holzapfel@tum.de>

import json
import logging
import logging.handlers
import os


class MsgCounterFileHandler(logging.Handler):
    """
    A handler class which counts the logging records by level and periodically writes the counts to a json file.
    """

    def __init__(self, filename, continue_counts=True, *args, **kwargs):
        """
        Initialize the handler.

        PARAMETER
        ---------
        continue_counts: bool, optional
            defines if the counts should be loaded and restored if the json file exists already.
        """
        logging.Handler.__init__(self, *args, **kwargs)

        filename = os.fspath(filename)
        self.baseFilename = os.path.abspath(filename)

        self.continue_counts = continue_counts

        # if another instance of this class is created, get the actual counts
        self.level2count_dict = self.load_counts_from_file()

        # set level and update the json file, else do it when the level is set
        if self.level is not logging.NOTSET:
            self.setLevel(self.level)

    def emit(self, record):
        """
        Counts a record.
        In case, create add the level to the dict.
        If the time has come, update the json file.
        """
        level_name = record.levelname
        if level_name not in self.level2count_dict:
            self.level2count_dict[level_name] = 0
        self.level2count_dict[level_name] += 1

        self.flush()

    def flush(self):
        """
        Flushes the dictionary.
        """
        self.acquire()
        try:
            with open(self.baseFilename, 'w') as f:
                json.dump(self.level2count_dict, f)
        finally:
            self.release()

    def setLevel(self, level):
        """
        Set the logging level of this handler.  level must be an int or a str.
        """
        # noinspection PyUnresolvedReferences
        # noinspection PyProtectedMemeber
        self.level = logging._checkLevel(level)

        # add levels to the counter dict. for the levels the Handler is listening to
        # noinspection PyUnresolvedReferences PyProtectedMemeber
        for level_int, level_name in logging._levelToName.items():  # get the dict, WARNING can be ignored
            if level_name not in self.level2count_dict and level_int >= self.level:
                self.level2count_dict[level_name] = 0

        self.flush()  # create the file

    def load_counts_from_file(self):
        """
        Load the dictionary from a json file or create an empty dictionary
        """
        level2count_dict = {}
        if os.path.exists(self.baseFilename) and self.continue_counts:
            try:
                with open(self.baseFilename) as f:
                    level2count_dict = dict(json.load(f))
            except Exception as a:
                logging.warning(f'Failed to load counts with: {a}')
                level2count_dict = {}

        return level2count_dict


def increasing_log_level(count, thresholds=None):
    """ Returns a log level depending on the count. This can be used to increase the level when an event occurs
    regularly and the number of events is counted.
    PARAMETER
    ---------
    count: int
        actual count of the event
    thresholds: None or list, optional
        Defines the thresholds. It has to be a list with length 4 and the thresholds are:
        [INFO, WARNING, ERROR, CRITICAL]. Default: [1, 3, 5, 7]"""
    if thresholds is None:
        thresholds = [1, 3, 5, 7]
    log_level = logging.DEBUG
    if count > thresholds[3]:
        log_level = logging.CRITICAL
    elif count > thresholds[2]:
        log_level = logging.ERROR
    elif count > thresholds[1]:
        log_level = logging.WARNING
    elif count > thresholds[0]:
        log_level = logging.INFO
    return log_level


def setup_logging(log_level=logging.INFO,
                  to_console=False,
                  file_name=None,
                  msg_counter_file=None,
                  msg_formatter_file=None,
                  msg_formatter_console=None):
    """ Setup of the logging module. This piece of code has to be placed at the very beginning of the 'main' code,
    right after the imports or even before some imports i.e. `scheduler` or `Pyro` if those modules should have a
    different logging level than the rest.

    PARAMETERS
    ----------
    log_level: logging level, optional
        The general logging level. Default is INFO.
        Can be one of [DEBUG, INFO, WARNING, ERROR, CRITICAL] (in increasing order) or other manually added levels.
    to_console: bool, logging.Level, optional
        If True (default False), logging prints to the console with the log_level specified in `log_level`.
        Alternatively a logging level can be specified which enables the logging to the console with this level.
    file_name: Str or None, optional
        If set, logging to a file is enabled. If None (default), no log file is written.
    msg_counter_file: Str or None, optional
        If set, the logging msg are counted and saved in the file defined. In case the ending is replaced to '.json'.
        This can be used to monitor the system state, i.e. with InfluxDB-Telegraf-Grafana.
    msg_formatter_file: str or None, optional
        To define the log-msg format, e.g.: '%(asctime)s;%(levelname)s;%(message)s' (Default with None).
        See below in the DocString for more information on the logging formatter
    msg_formatter_console: str or None, optional
        To define the log-msg format, e.g.: '%(asctime)s;%(levelname)7s;%(name)20s: %(message)s' (Default with None).
        See below in the DocString for more information on the logging formatter

    The formatter documentation is available here:
    `https://docs.python.org/3.1/library/logging.html#logging.Formatter` or
    `https://docs.python.org/3.1/library/logging.html#logging.handlers.HTTPHandler`.
    - For padding, you can add a min size, e.g. '%(funcName)20s'
    - A more detailed msg helpfully for debugging could be: '%(asctime)s;%(levelname)s;%(processName)s;
        %(threadName)s;%(thread)d;%(name)s;%(funcName)s;%(lineno)d;%(message)s'
    """
    log_level = log_level
    logging.getLogger('schedule').setLevel(logging.WARNING)

    if msg_formatter_file is None:
        msg_formatter_file = '%(asctime)s;%(levelname)s;%(message)s'

    if msg_formatter_console is None:
        msg_formatter_console = '%(asctime)s;%(levelname)7s;%(name)20s: %(message)s'

    # add SHUTDOWN level
    logging.addLevelName(60, "SHUTDOWN")
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    if file_name is not None:
        # create file handler which logs to a file. The files are rotated and kept for 10 rotations.
        # E.g.: 'W0' once per week on Monday (or Sunday?) a new file is started
        fh = logging.handlers.TimedRotatingFileHandler(file_name, when='W0', backupCount=10, utc=True)
        fh.setLevel(log_level)
        formatter = logging.Formatter(msg_formatter_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if to_console:
        # Create console handler, with a different formatter
        ch = logging.StreamHandler()
        if type(to_console) is bool:
            ch.setLevel(log_level)
        else:
            ch.setLevel(to_console)
        ch.setFormatter(logging.Formatter(msg_formatter_console))  # formatter)
        logger.addHandler(ch)

    if msg_counter_file is not None:
        # Create msg counter handler which counts the msg with a level equal or higher than WARNING.
        # The counts are stored in a json file. Which can be used to monitor the state with
        # e.g. InfluxDB/Telegraf/Grafana
        if msg_counter_file.endswith('.json'):
            pass
        elif len(msg_counter_file[-5:].rsplit('.', 1)) > 1:  # just check if there is a '.' in the last 5 elements
            msg_counter_file = msg_counter_file.rsplit('.', 1)[0] + '.json'  # replace the ending

        msg_counter_handler = MsgCounterFileHandler(msg_counter_file, level=logging.WARNING)
        logger.addHandler(msg_counter_handler)

    return logger

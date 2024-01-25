#!/usr/bin/python3

# Author: Kilian Holzapfel <kilian.holzapfel@tum.de>


# TODO: under construction

class DAQDaemonParser:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def status():
        return False

    @staticmethod
    def start():
        return ''

    @staticmethod
    def stop():
        return ''

    def handle_sdaq_cmdline(self, args, output_buffer):
        """
        Handles the logging for all pmtspec-module-type sensor values.
        """
        # get it safe against output_buffer=None
        if output_buffer is None:
            output_buffer = []

        if str(args.status).lower() in ["0", "off", "stop"]:
            message = self.stop()
        elif str(args.status).lower() in ["1", "on", "start"]:
            message = self.start()
        else:
            message = f'SDAQ-{self.name} is: {"active" if self.status() else "off"}'

        output_buffer.append(message)

    @staticmethod
    def create_subsys_argparser(subsys_parsers):
        # Module logging (sdaq).
        logging_parser = subsys_parsers.add_parser('sdaq',
                                                   help='Managing the scheduled DAQ (sdaq) of the module sensors.')
        logging_parser.set_defaults(cmd='sdaq')
        logging_parser.add_argument(
            'status', metavar='STATE', nargs='?', type=str,
            help='start/stop',
            default=None)

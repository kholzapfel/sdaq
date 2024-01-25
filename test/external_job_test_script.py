#!/usr/bin/env python3

import sys
import numpy as np
import time
import argparse

# A script for unittests with the ExternalJob

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('length', metavar='N', type=int, nargs='?', default=4,
                    help='the length of returned parameters')
parser.add_argument('--loops', default=1, type=int,
                    help='The number of loops the process will do. Negative values mean infinite loops.')
parser.add_argument('--delimiter', default=' ', type=str, help='delimiter of the values')
parser.add_argument('--type', default='int', type=str, nargs='+', help='type of the values')
parser.add_argument('--random_min', default=-10, type=float, help='min of the value range')
parser.add_argument('--random_max', default=10, type=float, help='max of the value range')
parser.add_argument('--period', default=.01, type=float, help='seconds between two loops')
parser.add_argument('--raise_error', action='store_true', help='if set, process raise an error when finished')
parser.add_argument('--provide_time', action='store_true', help='if set, it adds a timestamp as first item')
parser.add_argument('--random', action='store_true', help='if set, the returns are random')
parser.add_argument('--verbose', action='store_true', help='if set, the returns the parsed args')


args = parser.parse_args()
if args.verbose:
    print('#', args)

if __name__ == "__main__":
    i = args.loops
    start_time = time.time()
    next_time = start_time
    while i != 0:
        if args.random:
            data = np.random.random_sample(args.length)
            data = data * (args.random_max - args.random_min) + args.random_min
        else:
            data = np.arange(args.length) + (args.loops-i)

        data = data.astype(args.type)
        if args.provide_time:
            data = np.append([time.time()-start_time], data)

        print(args.delimiter.join(data.astype('str')), flush=True)
        i -= 1
        if i != 0:
            next_time += args.period
            time.sleep(next_time - time.time())

    if args.raise_error:
        raise RuntimeError(f'Process finished {args.loops}')

    sys.exit(0)

import shutil
import os
import sys
import argparse
import re
import numpy as np
from argparse import RawTextHelpFormatter
from glob import glob


# arguments
parser = argparse.ArgumentParser(
    description='''A module to copy big number of files''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Copy wisely''')
parser.add_argument('-i', '--input', default=os.getcwd(), help='directory to copy from. Default - WD')
parser.add_argument('--file', default=None, help='File name to copy. Default - NONE')
parser.add_argument('--rnd', default=None, type=float, help='Select random cells? Accepts float[0.0:1.0] Default = NONE')
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-o', '--output', help='Input file name', required=True)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":

    # create output dir
    if not os.path.exists(argsP.output):
        os.makedirs(argsP.output)

    # use selection ?
    if argsP.rnd != None:
        uniq_wells = np.unique([re.search(r'(Pt\d+_r\d+c\d+)', x)[0] for x in glob(os.path.join(argsP.input, '*tif'))])

        for i in uniq_wells:
            print("Copying file {}".format(i))
            sys.stdout.flush()
            all_files = glob(os.path.join(argsP.input, "".join([i, '*tif'])))
            rand_select = np.random.choice(all_files, round(len(all_files) * argsP.rnd), replace=False)

            for f in rand_select:
                shutil.copy(f, argsP.output)

    else:
        for f in glob(os.path.join(argsP.input, "".join([argsP.file, '*tif']))):
            shutil.copy(f, argsP.output)



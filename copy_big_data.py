import shutil

import os
import sys
import argparse
from argparse import RawTextHelpFormatter


# arguments
parser = argparse.ArgumentParser(
    description='''A module to copy big number of files''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Copy wisely''')
parser.add_argument('-i', '--input', default = os.getcwd(), help='directory to copy from. Default - WD')
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-o', '--output', help='Input file name', required=True)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



for f in os.listdir(argsP.input):
    if os.path.isfile(f):
        shutil.copy(f,argsP.output)

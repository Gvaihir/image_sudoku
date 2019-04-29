import shutil
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from glob import glob


# arguments
parser = argparse.ArgumentParser(
    description='''A module to copy big number of files''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Count files wisely''')
parser.add_argument('-i', '--input', default=os.getcwd(), help='directory to copy from. Default - WD')
parser.add_argument('--file', type=str, default="*", help='File name string to count. Default - *')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":
    print(len(glob(os.path.join(argsP.input, argsP.file))))

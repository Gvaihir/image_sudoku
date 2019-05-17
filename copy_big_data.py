import shutil
import os
import sys
import argparse
from argparse import RawTextHelpFormatter
from glob import glob
import pandas as pd


# arguments
parser = argparse.ArgumentParser(
    description='''A module to copy big number of files''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Copy wisely''')
parser.add_argument('-i', '--input', default=os.getcwd(), help='directory to copy from. Default - WD')
parser.add_argument('--file', default=None, help='File name to copy. Default - NONE')
parser.add_argument('--selection', default=None, type=str, help='STR Select specific portions of the screen. Accepts tab delimited list of plates/wells. See samplesheet. Default = NONE')
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('-o', '--output', help='Input file name', required=True)

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()



if __name__ == "__main__":

    # use selection ?
    if argsP.selection != None:
        selection_df = pd.read_csv(argsP.selection, sep='\t')
        selection_df['well_row'] = 'r' + selection_df['well'].replace(r'\d', '', regex=True).apply(
            lambda x: ord(x) - 64).apply(lambda x: format(x, '02d'))
        selection_df['well_col'] = 'c' + selection_df['well'].replace(r'\D', '', regex=True).astype(int).apply(
            lambda x: format(x, '02d'))
        selection_df['sample'] = selection_df['well_row'].str.cat(selection_df['well_col'])

        # subset dfDir using only sample names from selection
        dfDir = dfDir[dfDir["Sample_selection"].isin(selection_df['sample'])]

    for f in glob(os.path.join(argsP.input, "".join([argsP.file, '*tif']))):
        shutil.copy(f, argsP.output)



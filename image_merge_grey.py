import cv2
import numpy as np
import os
import pandas as pd
import glob
import sys
import argparse
from argparse import RawTextHelpFormatter
from image_bin import split_func
from pp_combine_export import pp_combine_export_grey

parser = argparse.ArgumentParser(
    description='''Program to merge fluorescent image channels and prepare for segmentation''',
    formatter_class=RawTextHelpFormatter,
    epilog="""Merge wisely""")
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--coeff', default = 1, type=float, help='FLOAT Pixel intensity coefficient: PI\' = 8bitPI * coeff (Default: 1)')
parser.add_argument('--channel', default = ch1, choices=['ch1', 'ch2', 'ch3'], help='STR Channel name which to extract. Accepted values: ch1, ch2, ch3')
parser.add_argument('--tile_size', default = 256, type=int, help='INT Size of a tile for splitting')
parser.add_argument('--ntiles', default = 4, type=int, help='INT Number of tiles to split into. Default = 4 (each image will be split into 4)')
parser.add_argument('--selection', default = None, type=str, help='STR Select specific portions of the screen. Accepts tab delimited list of plates/wells. See samplesheet. Default = NONE')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


# ignoring hidden files
def listdir_nohidden(path):
    return [os.path.basename(x) for x in glob.glob(os.path.join(path, '*'))]


if __name__ == "__main__":
    inPath = argsP.wd
    inDir = listdir_nohidden(inPath)
    inDir.sort()
    dfDir = pd.DataFrame(inDir, columns=['Name'])
    outPath = '/'.join([os.path.dirname(inPath), "converted_grey"])
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # make clahe function
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # split file name, extract sample name
    dfDir["Sample"] = dfDir["Name"].str.split('-', expand=True)[0]
    dfDir["Sample_selection"] = dfDir["Sample"].str.split('f', expand=True)[0]
    # use selection ?
    if argsP.selection != None:
        selection_df = pd.read_csv(argsP.selection, sep='\t')
        selection_df['well_row'] = 'r' + selection_df['well'].replace(r'\d', '', regex=True).apply(lambda x: ord(x)-64).apply(lambda x: format(x, '02d'))
        selection_df['well_col'] = 'c' + selection_df['well'].replace(r'\D', '', regex=True).astype(int).apply(lambda x: format(x, '02d'))
        selection_df['sample'] = selection_df['well_row'].str.cat(selection_df['well_col'])

        # subset dfDir using only sample names from selection
        dfDir = dfDir[dfDir["Sample_selection"].isin(selection_df['sample'])]



    # loop over each sample
    for num, x in enumerate(dfDir["Sample"].unique()):
        # subset df for relevant sample name
        dfRel = dfDir[dfDir.Sample == x]

        # import images for each channel
        b = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains(argsP.channel), 'Name'].to_string(index=False)]), -1)


        # work on different channels
        # perform split and correct function
        b = split_func(x=b, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)

        pp_combine_export_grey(b_list=b,
                          b_coeff=argsP.coeff,
                          outpath=outPath, imName=x)


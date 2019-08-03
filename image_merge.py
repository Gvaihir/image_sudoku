import cv2
import numpy as np
import os
import pandas as pd
import glob
import sys
import argparse
from argparse import RawTextHelpFormatter
from image_bin import split_func
from pp_combine_export import pp_combine_export
from pp_edge_detect import pp_edge_detect
from histo_match import hist_match

parser = argparse.ArgumentParser(
    description='''Program to merge fluorescent image channels and prepare for segmentation''',
    formatter_class=RawTextHelpFormatter,
    epilog="""Merge wisely""")
parser.add_argument('--wd', default = os.getcwd(), help='directory with images. Default - WD')
parser.add_argument('--ch_coeff', default = [1,1,1], nargs='+', type=float, help='FLOAT Pixel intensity coefficient: PI\' = 8bitPI * ch_coeff \n'
                                                                                 '(Default: BLUE: 1; GREEN: 1; RED: 1)')
parser.add_argument('--chs', default = [1,2,3], nargs='+', type=str, help='STR Channels to use')
parser.add_argument('--tile_size', default = 2048, type=int, help='INT Size of a tile for splitting')
parser.add_argument('--ntiles', default = 1, type=int, help='INT Number of tiles to split into. Default = 1 (each image will be split into 1)')
parser.add_argument('--format', default = 'tif', type=str, help='Image format. Default = TIF')
parser.add_argument('--name_pattern', default = "r'([AB-Z]\d+_s\d)'", type=str, help='String pattern for image name. Uses Regex Default = ([AB-Z]\d+_s\d)')
parser.add_argument('--ch_pattern', default = 'ch', type=str, help='String pattern which defines channel. Accepted {ch, w}. Default = CH')
parser.add_argument('--edge_red', default = False, type=bool, help='BOOL Option to perform LoG on red channel. Default = False')
parser.add_argument('--laplace_kernel', default = 3, type=int, help='INT Kernel for LoG. Has to be odd. Default = 3')
parser.add_argument('--selection', default = None, type=str, help='STR Select specific portions of the screen. Accepts tab delimited list of plates/wells. See samplesheet. Default = NONE')

if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


# ignoring hidden files
def listdir_nohidden(path):
    return [os.path.basename(x) for x in glob.glob(os.path.join(path, '*.'+argsP.format+'*'))]


if __name__ == "__main__":
    inPath = argsP.wd
    inDir = listdir_nohidden(inPath)
    inDir.sort()
    dfDir = pd.DataFrame(inDir, columns=['Name'])
    outPath = '/'.join([os.path.dirname(inPath), "converted_rgb"])
    if not os.path.exists(outPath):
        os.makedirs(outPath)

    # make clahe function
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # split file name, extract sample name
    dfDir["Sample"] = dfDir["Name"].str.extract(eval(argsP.name_pattern))
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
        b = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains(argsP.ch_pattern+str(argsP.chs[0])), 'Name'].to_string(index=False).strip()]), -1)
        g = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains(argsP.ch_pattern+str(argsP.chs[1])), 'Name'].to_string(index=False).strip()]), -1)
        r = cv2.imread('/'.join([inPath, dfRel.loc[dfRel.Name.str.contains(argsP.ch_pattern+str(argsP.chs[2])), 'Name'].to_string(index=False).strip()]), -1)


        # work on different channels
        # perform split and correct function
        b = split_func(x=b, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)
        g = split_func(x=g, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)
        r = split_func(x=r, ntiles=argsP.ntiles, tile_size=argsP.tile_size, clahe=clahe)


        # detect edges
        if argsP.edge_red:
            r = pp_edge_detect(x=r, ddepth=cv2.CV_16U, kernel_size=argsP.laplace_kernel)


        pp_combine_export(b_list=b, g_list=g, r_list=r,
                          b_coeff=argsP.ch_coeff[0], g_coeff=argsP.ch_coeff[1], r_coeff=argsP.ch_coeff[2],
                          outpath=outPath, imName=x)


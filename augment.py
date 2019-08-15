#!usr/bin/env pythonimport osimport sysimport argparsefrom argparse import RawTextHelpFormatterfrom glob import globimport cv2import numpy as npimport imgaug as iafrom imgaug import augmenters as iaa# imgaug random seedia.seed(4)np.random.seed(6)# argumentsparser = argparse.ArgumentParser(    description='''A module to augment the image data by rotation, flip and gaussian noise''',    formatter_class=RawTextHelpFormatter,    epilog='''Augment wisely''')parser.add_argument('--wd', default = None, help='directory with images separated into subdirectories with classes. Default - NONE')parser.add_argument('--n_aug', default = 5, type=int, help='Number of images to generate per one input image. Default = 5')parser.add_argument('--out', default=os.path.join(os.getcwd(), 'cropped'), help='output dir. Default - WD/cropped')if len(sys.argv)==1:    parser.print_help(sys.stderr)    sys.exit(1)argsP = parser.parse_args()def augmenter(x, outdir, outname, i):    """    Fucntion to apply augmentation i times and export every augmented image.    Arguments:        x - image array        outdir - output directory        outname - name of image        i - iteration    """    seq = iaa.Sequential([        iaa.Affine(rotate=[-90, 90, 180]),        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),        iaa.Fliplr(0.5),        iaa.Flipud(0.5)    ], random_order=True)    aug_img = seq.augment_image(x)    cv2.imwrite(os.path.join(outdir, "iter{}_".format(i) + outname), aug_img)if __name__ == "__main__":    subdirs = os.listdir(argsP.wd)    for subdir in subdirs:        if os.path.isdir(os.path.join(argsP.wd, subdir)):            images = glob(os.path.join(argsP.wd, subdir, '*.tif*'))            for image in images:                img = cv2.imread(image, -1)                outname = os.path.basename(image)                outdir = os.path.join(argsP.out, subdir)                if not os.path.exists(outdir):                    os.makedirs(outdir)                [augmenter(x=img, outdir=outdir, outname=outname, i=i) for i in range(argsP.n_aug)]        else:            continue
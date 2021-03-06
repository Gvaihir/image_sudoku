# Copyright (c) 2018-present, Anton (Gvaihir) Ogorodnikov, Ye lab UCSF.

from __future__ import print_function

try:
    raw_input
except:
    raw_input = input

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os
import sys
import time

# keras
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam

# metrics
from sklearn.metrics import mean_squared_error as mse
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon as jsd



# logging
import wandb

# output
import json



parser = argparse.ArgumentParser(
    description='''Deployment of ACAE for data filtering. Outputs JSON with image, full path and adversarial and reconstruction losses''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Filter wisely''')

# Main parameters
parser.add_argument('-i', '--img_wd', default=None, help='directory with images organized as dir/subdir/image. Default - NONE')
parser.add_argument('-m', '--models', default=None, help='directory with models organized as dir/model.h5. Default - NONE')
parser.add_argument('-b', '--batch', default=256, type=int, help='Batch size')
parser.add_argument('-o', '--out', default=None, help='output dir organized -o/file.json. Default - NONE')
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')



if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

def model_compile(models):
    '''
    Loads and compiles models
    :param models: directory with encoder, decoder and discriminator

    :return: compiled autoencoder, discriminator and dimensions
    '''

    encoder = load_model(os.path.join(models, "encoder.h5"))
    decoder = load_model(os.path.join(models, "decoder.h5"))

    # get input and latent space shapes
    latent_dim = encoder.get_layer(index=-1).output_shape[1]
    input_dim = encoder.get_layer(index=0).input_shape[1:]

    # compile imported models into assembled autoencoder
    autoencoder_input = Input(shape=input_dim)
    autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error", metrics=['accuracy'])

    return autoencoder, encoder, input_dim, latent_dim


def mse_batch(data_x, data_y, input_dim):
    '''
    Function to compute reconstruction (autoencoder) loss as MSE
    :param data_x: input image
    :param data_y: reconstructed image
    :param input_dim: input dimensions
    :return: mse
    '''
    shape_1 = input_dim[0]
    shape_2 = input_dim[1:]
    reshape_x = np.reshape(data_x, (shape_1, np.prod(shape_2)))
    reshape_y = np.reshape(data_y, (shape_1, np.prod(shape_2)))
    mse_out = mse(reshape_x, reshape_y)
    return round(float(mse_out), 7)



def jsd_batch(latent_x, latent_dim):
    '''
    Function to compute adversarial loss
    :param latent_x: encoded latent space
    :return: Jensen-Shannon divergence b/w prior
    '''
    prior = np.random.randn(latent_dim) * 5.
    jsd_out = jsd(norm.pdf(latent_x), norm.pdf(prior))
    return round(float(jsd_out), 7)

def ceildiv(a, b):
    '''
    Just a ceiling division
    '''
    return -(-a // b)


# Main class
class ACAE_prediction(object):
    """
    Obect with preditions from ACAE
    Keys:

    img_name - full image name
    ae_loss - autoencoder loss (MSE)
    adv_loss - js divergence
    """

    def __init__(self):
        '''
        creates set of keys
        :param models: path to models
        '''


        self.image = []
        self.ae_loss = []
        self.adv_loss = []


    def anomaly_score(self, img_wd, batch):
        '''
        function that calculates anomaly scores

        :param img_wd: directory with images
        :param batch: batch size
        :param input_dim: dimensions of input vector
        :return:
        '''
        data_loader = ImageDataGenerator(
            rescale=1. / 255,
            featurewise_center=True,
            featurewise_std_normalization=True,
            shear_range=0,
            zoom_range=0,
            horizontal_flip=False)

        data_in = data_loader.flow_from_directory(
            img_wd,
            target_size=(input_dim[0], input_dim[0]),
            batch_size=batch,
            class_mode='input',
            shuffle=False)

        self.image.extend(data_in.filepaths)

        if argsP.verbose:
            print("Data loaded")
            sys.stdout.flush()

        n_batches = ceildiv(len(data_in.filepaths), batch)
        batch_index = 0
        while batch_index <= data_in.batch_index:
            data = data_in.next()
            data_list = data[0]

            # reconstruction of images
            ae_pred = autoencoder.predict_on_batch(data_list)
            recons_mse = [mse_batch(data_list[x], ae_pred[x], input_dim) for x in range(len(ae_pred))]
            self.ae_loss.extend(recons_mse)

            # creating latent representation of image features
            fake_latent = encoder.predict(data_list)
            adv_loss = [jsd_batch(x, latent_dim) for x in fake_latent]
            self.adv_loss.extend(adv_loss)

            if argsP.verbose:
                print("Batch {}/{} finished".format(batch_index, n_batches))
                sys.stdout.flush()

            batch_index += 1


### MAIN PROGRAM

if __name__ == "__main__":

    global argsP, autoencoder, encoder, input_dim, latent_dim
    argsP = parser.parse_args()
    since = time.time()
    # compile models
    autoencoder, encoder, input_dim, latent_dim = model_compile(argsP.models)

    if argsP.verbose:
        print("Models compiled")
        sys.stdout.flush()

    obj = ACAE_prediction()
    obj.anomaly_score(img_wd=argsP.img_wd, batch=argsP.batch)

    # create output dir and file
    out_file = ".".join([os.path.basename(argsP.img_wd), "json"])

    ### Export JSON ###
    with open(os.path.join(argsP.out, out_file), "w") as file:
        json.dump(obj.__dict__, file)

    time_elapsed = time.time() - since
    print('Prediction complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))





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

# keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.optimizers import Adam

# logging
import wandb

# plotting and other
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
from sklearn.manifold import TSNE
from absl import flags

parser = argparse.ArgumentParser(
    description='''Adversarial autoencoder from Makhzani, Alireza, et al. "Adversarial autoencoders." arXiv preprint arXiv:1511.05644 (2015)''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Encode wisely''')

# Main parameters
parser.add_argument('-i', '--img_wd', default = None, help='directory with images. Default - NONE')
parser.add_argument('-s', '--sobel', action='store_true', help='Apply sobel transformation')
parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('-b', '--batch', default=256, type=int, help='Batch size')
parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'aae_model'), help='output dir. Default - WD/aae')
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')
parser.add_argument('--input_dim', default=[144, 144, 3], nargs='+', type=int, help='Dimensionality of an input image')
parser.add_argument('--latent_dim', default=128, type=int, help='Dimensionality of a latent space')


# Running modes
parser.add_argument('--train', action='store_true', help='Training mode of AAE')
parser.add_argument('--recons', action='store_true', help='Reconstructing mode of AAE')
parser.add_argument('--generate', action='store_true', help='Image generation mode from latent space')
parser.add_argument('--adversarial', action='store_true', help='Use adversarial model')
parser.add_argument('--itsr', action='store_true', help='Use ITSR variation of adversarial model')
parser.add_argument('--plot', action='store_true', help='Plot latent space')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


def create_model(latent_dim, verbose=False, save_graph=False,
                 adversarial=True):
    '''
    Creates model
    :param input_dim: tuple, dmensions of an image (w*h*ch). W and H has to give modulo of division by 8 = 0
    :param latent_dim: int, number of latent dimensions
    :param verbose: bool, chatty
    :param save_graph: bool, saves latent representation. Work only for 2d latent
    :param adversarial: bool, make adversarial model
    :return: autoencoder, (discriminator), (generator), encoder, decoder
    '''


    input_dim = (224, 224, 3)
    autoencoder_input = Input(shape=input_dim)
    generator_input = Input(shape=input_dim)


    ## ENCODER
    encoder = Sequential()

    if argsP.sobel:
        # Layer 1
        encoder.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), input_shape=input_dim,
                           data_format="channels_last"))
        encoder.add(Conv2D(2, kernel_size=(3, 3), strides=(1, 1), padding='same'))
        encoder.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                           data_format="channels_last"))
    else:
        # Layer 1
        encoder.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), input_shape=input_dim, activation='relu',
                           data_format="channels_last"))





    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))
    encoder.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    # Layer 2
    encoder.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))
    encoder.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Layer 3
    encoder.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))

    # Layer 4
    encoder.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))

    # Layer 5
    encoder.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))
    encoder.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same"))

    # Dense
    encoder.add(Flatten())
    encoder.add(Dropout(rate=0.5))
    encoder.add(Dense(4096, activation='relu'))
    encoder.add(Dropout(rate=0.5))
    encoder.add(Dense(4096, activation='relu'))
    encoder.add(BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5))
    encoder.add(Dense(latent_dim, activation=None))

    ## DECODER
    # Dense
    decoder = Sequential()
    decoder.add(Dense(4096, input_shape=(latent_dim,), activation='relu'))
    decoder.add(Dropout(rate=0.5))
    decoder.add(Dense(4096, activation='relu'))
    decoder.add(Dropout(rate=0.5))
    decoder.add(Dense(9216, activation='relu'))

    # Conv
    decoder.add(Reshape((6, 6, 256)))
    decoder.add(UpSampling2D((2, 2)))
    decoder.add(Conv2DTranspose(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    decoder.add(Conv2DTranspose(384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    decoder.add(Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu'))
    decoder.add(Conv2DTranspose(96, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu'))
    decoder.add(Conv2DTranspose(3, kernel_size=(12, 12), strides=(4, 4), padding='valid', activation='sigmoid'))



    if adversarial:
        discriminator = Sequential()
        discriminator.add(Dense(4096, input_shape=(latent_dim,), activation='relu'))
        discriminator.add(Dense(reshape_dim**2*16, activation='relu'))
        discriminator.add(Dense(1, activation='sigmoid'))



    autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error", metrics=['accuracy'])

    if adversarial:
        discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        discriminator.trainable = False
        generator = Model(generator_input, discriminator(encoder(generator_input)))
        generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

    if verbose:
        print("Autoencoder Architecture")
        print(autoencoder.summary())
        if adversarial:
            print("Discriminator Architecture")
            print(discriminator.summary())
            print("Generator Architecture")
            print(generator.summary())

    if save_graph:
        plot_model(autoencoder, to_file="autoencoder_graph.png")
        if adversarial:
            plot_model(discriminator, to_file="discriminator_graph.png")
            plot_model(generator, to_file="generator_graph.png")

    if adversarial:
        return autoencoder, discriminator, generator, encoder, decoder
    else:
        return autoencoder, None, None, encoder, decoder









def train(train_data, out, latent_dim, n_epochs, autoencoder, discriminator, generator, encoder, decoder,
          adversarial = True):
    '''
    Function to train autoencoder. Arguments will be taken from argparse
    :param train_data: input data from flow_from_directory
    :param out: dir to save the models
    :param latent_dim: number of latent dimensions
    :param n_epochs: Number of epochs
    :param autoencoder: created autoencoder model
    :param discriminator: created discriminator model
    :param generator: created generator model
    :param encoder: created encoder part of autoencoder
    :param decoder: created decoder part of autoencoder
    :param adversarial: make adversarial model
    :return: trained encoder, decoder, discriminator and generator
    '''

    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        if adversarial:
            discriminator_losses = []
            generator_losses = []

        autoencoder_history = autoencoder.fit_generator(train_data, epochs=1)

        if adversarial:
            batch_index = 0
            discriminator_batch_losses = []
            generator_batch_losses = []
            while batch_index <= train_data.batch_index:
                data = train_data.next()
                data_list = data[0]
                data_size = len(data_list)

                fake_latent = encoder.predict(data_list)
                discriminator_input = np.concatenate((fake_latent, np.random.randn(data_size, latent_dim) * 5.))
                discriminator_labels = np.concatenate((np.zeros((data_size, 1)), np.ones((data_size, 1))))
                discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1,
                                                          batch_size=data_size, validation_split=0.0, verbose=0)
                generator_history = generator.fit(data_list, y=np.ones((data_size, 1)), epochs=1,
                                                  batch_size=data_size, validation_split=0.0, verbose=0)
                batch_index = batch_index + 1
                discriminator_batch_losses.append(discriminator_history.history["loss"])
                generator_batch_losses.append(generator_history.history["loss"])


        autoencoder_losses.append(autoencoder_history.history["loss"])
        # WandB logging
        if adversarial:
            discriminator_losses.append(np.mean(discriminator_batch_losses))
            generator_losses.append(np.mean(generator_batch_losses))

            print("generator_loss = {}\n"
                  "generator_acc = {}".format(
                generator_history.history["loss"],
                generator_history.history["acc"]
            ))

            print("EPOCH {} DONE".format(epoch))

            # WandB logging
            wandb.log({"phase": epoch,
                       "ae_train_loss": autoencoder_history.history["loss"],
                       "ae_train_acc": autoencoder_history.history["acc"],
                       "gen_train_loss": generator_history.history["loss"],
                       "gen_train_acc": generator_history.history["acc"]}, step=epoch)
        else:
            wandb.log({"phase": epoch,
                       "ae_train_loss": autoencoder_history.history["loss"],
                       "ae_train_acc": autoencoder_history.history["acc"]}, step=epoch)


        if epoch % 50 == 0:
            print("\nSaving models...")
            encoder.save(os.path.join(out, 'encoder.h5'))
            decoder.save(os.path.join(out, 'decoder.h5'))
            if adversarial:
                discriminator.save(os.path.join(out, 'discriminator.h5'))
                generator.save(os.path.join(out, 'generator.h5'))

    encoder.save(os.path.join(out, 'encoder.h5'))
    decoder.save(os.path.join(out, 'decoder.h5'))
    if adversarial:
        discriminator.save(os.path.join(out, 'discriminator.h5'))
        generator.save(os.path.join(out, 'generator.h5'))



if __name__ == "__main__":

    # initialize monitoring with WandB
    wandb.init(config=argsP)
    wandb.config.update(argsP)  # adds all of the arguments as config variables

    # input_dim make tuple
    input_dim = tuple(argsP.input_dim)
    # CREATE MODELS
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=input_dim,
                                                                           latent_dim=argsP.latent_dim,
                                                                           verbose=argsP.verbose, save_graph=False,
                                                                           adversarial=argsP.adversarial
                                                                           )
    # LOAD DATA
    data_loader = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_data = data_loader.flow_from_directory(
        argsP.img_wd,
        target_size=(input_dim[0], input_dim[0]),
        batch_size=argsP.batch,
        class_mode='input')

    # training mode
    if argsP.train:
        train(train_data=train_data, out=argsP.out,
              latent_dim=argsP.latent_dim, n_epochs=argsP.epoch,
              autoencoder=autoencoder, discriminator=discriminator,
              generator=generator, encoder=encoder, decoder=decoder,
              adversarial=argsP.adversarial
              )


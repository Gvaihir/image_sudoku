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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Flatten, BatchNormalization
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
    description='''Adversarial convolutional autoencoder (ACAE) training''',
    formatter_class=RawTextHelpFormatter,
    epilog='''Encode wisely''')

# Main parameters
parser.add_argument('-i', '--img_wd', default = None, help='directory with images. Default - NONE')
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
parser.add_argument('--conv', action='store_true', help='Use convolutional model')
parser.add_argument('--itsr', action='store_true', help='Use ITSR variation of adversarial model')
parser.add_argument('--plot', action='store_true', help='Plot latent space')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


def create_model(input_dim, latent_dim, verbose=False, save_graph=False, conv=True,
                 adversarial=True):
    '''
    Creates model
    :param input_dim: tuple, dmensions of an image (w*h*ch). W and H has to give modulo of division by 8 = 0
    :param latent_dim: int, number of latent dimensions
    :param verbose: bool, chatty
    :param save_graph: bool, saves latent representation. Work only for 2d latent
    :param conv: bool, make convolutional model
    :param adversarial: bool, make adversarial model
    :return: autoencoder, (discriminator), (generator), encoder, decoder
    '''

    assert input_dim[0]%8 == 0, "Dimension error: Chose H and W dimensions that can be divided by 8 without remnant"
    autoencoder_input = Input(shape=input_dim)
    generator_input = Input(shape=input_dim)

    reshape_dim = int(input_dim[0] / (2 ** 3))
    if conv:
        # Assemble convolutional model
        encoder = Sequential()
        encoder.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_dim, padding='same', activation='relu',
                           data_format="channels_last"))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Flatten())
        encoder.add(Dense(reshape_dim**2*16, activation='relu'))  # different, was 256
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Dense(reshape_dim**2*16, activation='relu'))  # different, didn't exist
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Dense(latent_dim, activation=None))  # different (was reshaping to 64D)

        decoder = Sequential()
        decoder.add(Dense(reshape_dim**2*16, input_shape=(latent_dim,), activation='relu'))
        decoder.add(Dense(reshape_dim**2*16, activation='relu'))
        decoder.add(Dense(reshape_dim**2*32, activation='relu'))
        decoder.add(Reshape((reshape_dim, reshape_dim, 32)))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), padding='same', activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), padding='same', activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(3, kernel_size=(5, 5), padding='same', activation='sigmoid')) # Relu in CellCognition

        if adversarial:
            discriminator = Sequential()
            discriminator.add(Dense(reshape_dim**2*16, input_shape=(latent_dim,), activation='relu'))
            discriminator.add(Dense(reshape_dim**2*16, activation='relu'))
            discriminator.add(Dense(1, activation='sigmoid'))



#TODO: non-convolutional
    else:
        encoder = Sequential()
        encoder.add(Dense(reshape_dim**2*16, input_shape=input_dim, activation='relu'))
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Dense(reshape_dim**2*16, activation='relu'))
        encoder.add(BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001))
        encoder.add(Dense(latent_dim, activation=None))

        decoder = Sequential()
        decoder.add(Dense(reshape_dim**2*16, input_shape=(latent_dim,), activation='relu'))
        decoder.add(Dense(reshape_dim**2*16, activation='relu'))
        decoder.add(Reshape((reshape_dim, reshape_dim, 3)))
        decoder.add(Dense(input_dim, activation='sigmoid'))

        if adversarial:
            discriminator = Sequential()
            discriminator.add(Dense(1600, input_shape=(latent_dim,), activation='relu'))
            discriminator.add(Dense(1000, activation='relu'))
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

        # WandB logging
        if adversarial:
            discriminator_loss = np.mean(discriminator_batch_losses)
            generator_loss = np.mean(generator_batch_losses)

            print("discriminator_loss = {}\n".format(
                discriminator_loss
            ))

            print("EPOCH {} DONE".format(epoch))

            # WandB logging
            wandb.log({"phase": epoch,
                       "ae_train_loss": autoencoder_history.history["loss"],
                       "ae_train_acc": autoencoder_history.history["acc"],
                       "discr_train_loss": discriminator_loss,
                       "gen_train_loss": generator_loss}, step=epoch)
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


'''
def reconstruct(n_samples):
    #encoder = load_model('{}_encoder.h5'.format(desc))
    #decoder = load_model('{}_decoder.h5'.format(desc))

    choice = np.random.choice(np.arange(n_samples))
    original = x_test[choice].reshape(1, 784)
    normalize = colors.Normalize(0., 255.)
    original = normalize(original)
    latent = encoder.predict(original)
    reconstruction = decoder.predict(latent)
    draw([{"title": "Original", "image": original}, {"title": "Reconstruction", "image": reconstruction}])
'''
# TODO: contionue training

# TODO: all the following
'''
def reconstruct(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    decoder = load_model('{}_decoder.h5'.format(desc))

    choice = np.random.choice(np.arange(n_samples))
    original = x_test[choice].reshape(1, 784)
    normalize = colors.Normalize(0., 255.)
    original = normalize(original)
    latent = encoder.predict(original)
    reconstruction = decoder.predict(latent)
    draw([{"title": "Original", "image": original}, {"title": "Reconstruction", "image": reconstruction}])


def generate(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    if latent is None:
        latent = np.random.randn(1, FLAGS.latent_dim)
    else:
        latent = np.array(latent)
    sample = decoder.predict(latent.reshape(1, FLAGS.latent_dim))
    draw([{"title": "Sample", "image": sample}])


def draw(samples):
    fig = plt.figure(figsize=(5 * len(samples), 5))
    gs = gridspec.GridSpec(1, len(samples))
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    raw_input("Press Enter to Exit")


def generate_grid(latent=None):
    decoder = load_model('{}_decoder.h5'.format(desc))
    samples = []
    for i in np.arange(400):
        latent = np.array([(i % 20) * 1.5 - 15., 15. - (i / 20) * 1.5])
        samples.append({
            "image": decoder.predict(latent.reshape(1, FLAGS.latent_dim))
        })
    draw_grid(samples)


def draw_grid(samples):
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(20, 20, wspace=-.5, hspace=0)
    for i, sample in enumerate(samples):
        ax = plt.Subplot(fig, gs[i])
        ax.imshow((sample["image"] * 255.).reshape(28, 28), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        # ax.set_title(sample["title"])
        fig.add_subplot(ax)
    plt.show(block=False)
    raw_input("Press Enter to Exit")


# fig.savefig("images/{}_grid.png".format(desc), bbox_inches="tight", dpi=300)

def plot(n_samples):
    encoder = load_model('{}_encoder.h5'.format(desc))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = x_test[:n_samples].reshape(n_samples, 784)
    y = y_test[:n_samples]
    normalize = colors.Normalize(0., 255.)
    x = normalize(x)
    latent = encoder.predict(x)
    if argsP.latent_dim > 2:
        tsne = TSNE()
        print("\nFitting t-SNE, this will take awhile...")
        latent = tsne.fit_transform(latent)
    fig, ax = plt.subplots()
    for label in np.arange(10):
        ax.scatter(latent[(y_test == label), 0], latent[(y_test == label), 1], label=label, s=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.set_aspect('equal')
    ax.set_title("Latent Space")
    plt.show(block=False)
    raw_input("Press Enter to Exit")


# fig.savefig("images/{}_latent.png".format(desc), bbox_inches="tight", dpi=300)
'''

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
                                                                           conv=argsP.conv,
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


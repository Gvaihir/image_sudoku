from __future__ import print_function

try:
    raw_input
except:
    raw_input = input

import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import os

# kears
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Conv2DTranspose, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.optimizers import Adam


from absl import app

# logging
import wandb
from datetime import datetime

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
parser.add_argument('-e', '--epoch', default=100, type=int, help='Number of training epochs')
parser.add_argument('-b', '--batch', default=256, type=int, help='Batch size')
parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'aae_model'), help='output dir. Default - WD/aae')
parser.add_argument('-v', '--verbose', action='store_true', help='Image generation mode from latent space')

# Running modes
parser.add_argument('--train', action='store_true', help='Training mode of AAE')
parser.add_argument('--recons', action='store_true', help='Reconstructing mode of AAE')
parser.add_argument('--generate', action='store_true', help='Image generation mode from latent space')
parser.add_argument('--adversarial', action='store_true', help='Use adversarial model')
parser.add_argument('--conv', action='store_true', help='Use convolutional model. Arch from CellCognition')
parser.add_argument('--itsr', action='store_true', help='Use ITSR variation of adversarial model')
parser.add_argument('--plot', action='store_true', help='Plot latent space')
parser.add_argument('--latent_dim', default=2, type=int, help='Dimensionality of a latent space')

# Other
parser.add_argument('--latent_vec', default=None, help='Latent vector (use with --generate flag)')


if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
argsP = parser.parse_args()


def create_model(input_dim, latent_dim, verbose=False, save_graph=False):
    '''
    Creates model
    :param latent_dim: int, number of latent dimensions
    :param verbose: bool, chatty
    :param save_graph: bool, saves latent representation. Work only for 2d latent
    :return: autoencoder, (discriminator), (generator), encoder, decoder
    '''

    autoencoder_input = Input(shape=(80,80,3))
    generator_input = Input(shape=(80,80,3))

    # monitoring with WandB
    wandb.init(config=argsP)
    wandb.config.update(argsP)  # adds all of the arguments as config variables

    if argsP.conv:
        # Assemble convolutional model
        encoder = Sequential()
        encoder.add(Conv2D(16, kernel_size=(5,5), input_shape=(80,80,3), padding='same', activation='relu', data_format="channels_last"))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
        encoder.add(MaxPooling2D(pool_size=(2, 2)))
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Flatten())
        encoder.add(Dense(1000, activation='relu')) # different, was 256
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Dense(1000, activation='relu')) # different, didn't exist
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Dense(latent_dim, activation=None)) # different (was reshaping to 64D)

        decoder = Sequential()
        decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dense(1600, activation='relu'))
        decoder.add(Reshape((10,10,16)))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(16, kernel_size=(3, 3), padding='same', activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(16, kernel_size=(3, 3), padding='same', activation='relu'))
        decoder.add(UpSampling2D((2, 2)))
        decoder.add(Conv2DTranspose(3, kernel_size=(5, 5), padding='same', activation='sigmoid')) # Relu in CellCognition

        if argsP.adversarial:
            discriminator = Sequential()
            discriminator.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
            discriminator.add(Dense(1000, activation='relu'))
            discriminator.add(Dense(1600, activation='relu'))
            discriminator.add(Reshape((10, 10, 16)))
            discriminator.add(UpSampling2D((2, 2)))
            discriminator.add(Conv2DTranspose(16, kernel_size=(3, 3), padding='same', activation='relu'))
            discriminator.add(UpSampling2D((2, 2)))
            discriminator.add(Conv2DTranspose(16, kernel_size=(3, 3), padding='same', activation='relu'))
            discriminator.add(UpSampling2D((2, 2)))
            discriminator.add(
                Conv2DTranspose(3, kernel_size=(5, 5), padding='same', activation='sigmoid'))  # Relu in CellCognition




    else:
        encoder = Sequential()
        encoder.add(Dense(1000, input_shape=input_dim, activation='relu'))
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(BatchNormalization, axis=-1, momentum=0.9, epsilon=0.001)
        encoder.add(Dense(latent_dim, activation=None))

        decoder = Sequential()
        decoder.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
        decoder.add(Dense(1600, activation='relu'))
        decoder.add(Reshape((16, 10, 10)))
        decoder.add(Dense(input_dim, activation='sigmoid'))

        if argsP.adversarial:
            discriminator = Sequential()
            discriminator.add(Dense(1000, input_shape=(latent_dim,), activation='relu'))
            discriminator.add(Dense(1000, activation='relu'))
            discriminator.add(Dense(1, activation='sigmoid'))

    autoencoder = Model(autoencoder_input, decoder(encoder(autoencoder_input)))
    autoencoder.compile(optimizer=Adam(lr=1e-4), loss="mean_squared_error", metrics=['accuracy'])

    if argsP.adversarial:
        discriminator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])
        discriminator.trainable = False
        generator = Model(generator_input, discriminator(encoder(generator_input)))
        generator.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=['accuracy'])

    if verbose:
        print("Autoencoder Architecture")
        print(autoencoder.summary())
        if argsP.adversarial:
            print("Discriminator Architecture")
            print(discriminator.summary())
            print("Generator Architecture")
            print(generator.summary())

    if save_graph:
        plot_model(autoencoder, to_file="autoencoder_graph.png")
        if argsP.adversarial:
            plot_model(discriminator, to_file="discriminator_graph.png")
            plot_model(generator, to_file="generator_graph.png")

    if argsP.adversarial:
        return autoencoder, discriminator, generator, encoder, decoder
    else:
        return autoencoder, None, None, encoder, decoder


def train(wd, batch_size, latent_dim, n_epochs):
    autoencoder, discriminator, generator, encoder, decoder = create_model(input_dim=argsP.input_dim, latent_dim=argsP.latent_dim)
    data_loader = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_data = data_loader.flow_from_directory(
        wd,
        target_size=(80, 80),
        batch_size=batch_size,
        class_mode='input')

    past = datetime.now()
    for epoch in np.arange(1, n_epochs + 1):
        autoencoder_losses = []
        if argsP.adversarial:
            discriminator_losses = []
            generator_losses = []

            autoencoder_history = autoencoder.fit_generator(train_data, epochs=1)

            if argsP.adversarial:
                fake_latent = encoder.predict_generator(train_data)
                discriminator_input = np.concatenate((fake_latent, np.random.randn(batch_size, latent_dim) * 5.))
                discriminator_labels = np.concatenate((np.zeros((batch_size, 1)), np.ones((batch_size, 1))))
                discriminator_history = discriminator.fit(x=discriminator_input, y=discriminator_labels, epochs=1,
                                                          batch_size=batch_size, validation_split=0.0, verbose=0)
                generator_history = generator.fit_generator(train_data, y=np.ones((batch_size, 1)), epochs=1,
                                                  batch_size=batch_size, validation_split=0.0, verbose=0)

            autoencoder_losses.append(autoencoder_history.history["loss"])
            if argsP.adversarial:
                discriminator_losses.append(discriminator_history.history["loss"])
                generator_losses.append(generator_history.history["loss"])

            # WandB logging
            wandb.log({"phase": epoch,
                       "ae_train_loss": autoencoder_history.history["loss"],
                       "ae_train_acc": autoencoder_history.history["acc"],
                       "ae_val_loss": autoencoder_history.history["val_loss"],
                       "ae_val_acc": autoencoder_history.history["val_acc"],

                       "gen_train_loss": generator_history.history["loss"],
                       "gen_train_acc": generator_history.history["acc"],
                       "gen_val_loss": generator_history.history["val_loss"],
                       "gen_val_acc": generator_history.history["val_acc"]}, step=epoch)


        if epoch % 50 == 0:
            print("\nSaving models...")
            # autoencoder.save('{}_autoencoder.h5'.format(desc))
            encoder.save(os.path.join(argsP.out, 'encoder.h5'))
            decoder.save(os.path.join(argsP.out, 'decoder.h5'))
        if argsP.adversarial:
            discriminator.save(os.path.join(argsP.out, 'discriminator.h5'))
            generator.save(os.path.join(argsP.out, 'generator.h5'))

    encoder.save(os.path.join(argsP.out, 'encoder.h5'))
    decoder.save(os.path.join(argsP.out, 'decoder.h5'))
    if argsP.adversarial:
        discriminator.save(os.path.join(argsP.out, 'discriminator.h5'))
        generator.save(os.path.join(argsP.out, 'generator.h5'))




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


def main(argv):
    global desc
    if argsP.adversarial:
        desc = "aae"
    else:
        desc = "regular"
    if argsP.train:
        train(n_samples=argsP.train_samples, batch_size=argsP.batchsize, n_epochs=argsP.epochs)
    elif argsP.reconstruct:
        reconstruct(n_samples=FLAGS.test_samples)
    elif argsP.generate:
        if argsP.latent_vec:
            assert len(
                argsP.latent_vec) == argsP.latent_dim, "Latent vector provided is of dim {}; required dim is {}".format(
                len(argsP.latent_vec), argsP.latent_dim)
            generate(argsP.latent_vec)
        else:
            generate()
    elif argsP.generate_grid:
        generate_grid()
    elif argsP.plot:
        plot(argsP.test_samples)


if __name__ == "__main__":
    app.run(main)
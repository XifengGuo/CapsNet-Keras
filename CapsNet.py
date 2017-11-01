"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python CapsNet.py
       
Result:
    validation accuracy > 99.4% after 10 epochs.
    About 5 minutes per epoch on a single GTX1070 card
    
Author: Xifeng Guo, E-mail: guoxifeng1990@163.com, Github: https://github.com/XifengGuo/CapsNet-Keras
"""
from keras import layers, models
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)

    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing)(primarycaps)

    out_caps = Length(name='out_caps')(digitcaps)

    # Reconstruction network
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[28, 28, 1], name='out_recon')(x_recon)

    return models.Model([x, y], [out_caps, x_recon])


def margin_loss(y_true, y_pred):
    """
    
    :param y_true: [None,
    :param y_pred: [None, num_capsule, dim_vector]
    :return: 
    """
    # [batch_size, num_capsule]
    # y_pred = K.sqrt(K.sum(K.square(y_pred), -1))
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import numpy as np
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks
    from keras.utils.vis_utils import plot_model

    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, x_batch])

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lam_recon', default=0.0005, type=float)
    parser.add_argument('--num_routing', default=3, type=int)
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--save_dir', default='./result')
    args = parser.parse_args()
    print(args)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir+'/tensorboard-logs', batch_size=args.batch_size)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', save_best_only=True, verbose=1)

    # define model
    model = CapsNet(input_shape=[28, 28, 1],
                    n_class=len(np.unique(np.argmax(y_train, 1))),
                    num_routing=args.num_routing)
    model.summary()
    plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    model.compile(optimizer='adam',
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint])
    """

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, x_test]],
                        callbacks=[log, tb, checkpoint])
    model.save(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'trained_model.h5\'')

    from utils import plot_log
    plot_log(args.save_dir + '/log.png')


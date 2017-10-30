"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python CapsNet.py
       
Result:

"""
from keras import layers, models
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length


def CapsNet(input_shape, batch_size):
    x = layers.Input(input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)

    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    digitcaps = CapsuleLayer(num_capsule=10, dim_vector=16, batch_size=batch_size)(primarycaps)

    out = Length()(digitcaps)

    return models.Model(x, out)


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
    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    args = parser.parse_args()
    print(args)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model = CapsNet([28,28,1], batch_size=args.batch_size)
    model.summary()
    model.compile(optimizer='adam', loss=margin_loss, metrics=['acc'])

    # begin training
    model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[x_test, y_test])
    model.save('trained_model.h5')
    print('Trained model saved to \'trained_model.h5\'')
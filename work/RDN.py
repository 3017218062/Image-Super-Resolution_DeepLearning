import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def RDN(inputShape=(128, 128, 3), scale=2, d=16, c=8):
    def DenseBlock(x, num):
        filters = x.shape[-1]
        res = tf.identity(x)
        for _ in range(num):
            x = Conv2D(filters, 3, activation="relu", padding="same")(x)
            x = concatenate([res, x], axis=-1)
            res = tf.identity(x)
        return x

    def RDB(x, num):
        _x = tf.identity(x)
        _x = DenseBlock(_x, num)
        _x = Conv2D(x.shape[-1], 1)(_x)
        return Add()([x, _x])

    i = Input(inputShape)
    x = Conv2D(64, 3, padding="same")(i)
    _x = Conv2D(64, 3, padding="same")(x)
    if d > 1:
        Fs = []
        for _ in range(d):
            _x = RDB(_x, c)
            Fs.append(_x)
        _x = concatenate(Fs, axis=-1)
    else:
        _x = RDB(_x, c)
    _x = Conv2D(_x.shape[-1], 1)(_x)
    _x = Conv2D(64, 3, padding="same")(_x)
    x = Add()([x, _x])
    x = UpSampling2D(scale, interpolation="bilinear")(x)
    o = Conv2D(3, 3, padding="same")(x)

    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = RDN()

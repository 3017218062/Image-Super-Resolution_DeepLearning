import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def DRRN(inputShape=(128, 128, 3), scale=2, b=1, u=25):
    def RB(x, u=9):
        convLayer = Sequential([
            Conv2D(128, 3, activation="relu", padding="same"),
            Conv2D(128, 3, activation="relu", padding="same"),
        ])
        x = Conv2D(128, 3, activation="relu", padding="same")(x)
        _x = tf.identity(x)
        for _ in range(u):
            x = convLayer(x)
            x = Add()([x, _x])
        return x

    i = Input(inputShape)
    x = UpSampling2D(scale, interpolation="bilinear")(i)
    _x = tf.identity(x)
    for _ in range(b):
        x = RB(x, u=u)
    x = Conv2D(3, 3, padding="same")(x)
    o = Add()([x, _x])
    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = DRRN()

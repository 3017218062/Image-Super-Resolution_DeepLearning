import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def VDSR(inputShape=(128, 128, 3), scale=2, d=20):
    i = Input(inputShape)
    x = UpSampling2D(scale, interpolation="bilinear")(i)
    _x = Conv2D(64, 3, activation="relu", padding="same")(x)
    for _ in range(d - 2):
        _x = Conv2D(64, 3, activation="relu", padding="same")(_x)
    _x = Conv2D(3, 3, padding="same")(_x)
    o = Add()([x, _x])
    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = VDSR()

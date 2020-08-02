import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def FSRCNN(inputShape=(128, 128, 3), scale=2):
    i = Input(inputShape)
    x = Conv2D(56, 5, padding="same")(i)
    x = PReLU()(x)
    x = Conv2D(12, 1)(x)
    x = PReLU()(x)
    for _ in range(4):
        x = Conv2D(12, 3, padding="same")(x)
        x = PReLU()(x)
    x = Conv2D(56, 1)(x)
    x = PReLU()(x)
    # o = Conv2DTranspose(3, 9, strides=scale, padding="same")(x)
    x = UpSampling2D(scale, interpolation="bilinear")(x)
    o = Conv2D(3, 9, padding="same")(x)

    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = FSRCNN()

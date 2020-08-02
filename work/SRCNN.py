import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def SRCNN(inputShape=(128, 128, 3), scale=2):
    i = Input(inputShape)
    x = UpSampling2D(scale, interpolation="bilinear")(i)
    x = Conv2D(64, 9, activation="relu", padding="same")(x)
    x = Conv2D(32, 1, activation="relu")(x)
    o = Conv2D(3, 5, padding="same")(x)

    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = SRCNN()

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def ESPCN(inputShape=(128, 128, 3), scale=2):
    i = Input(inputShape)
    x = Conv2D(64, 5, activation="tanh", padding="same")(i)
    x = Conv2D(32, 3, activation="tanh", padding="same")(x)
    x = Conv2D(3 * (scale ** 2), 3, padding="same")(x)
    o = Lambda(tf.nn.depth_to_space, arguments={"block_size": scale}, name="PixelShuffle")(x)

    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = ESPCN()

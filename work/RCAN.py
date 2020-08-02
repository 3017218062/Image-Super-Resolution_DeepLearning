import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def RCNN(inputShape=(128, 128, 3), scale=2, g=10, b=20):
    def SEBlock(x, reduction=16):
        _x = GlobalAveragePooling2D()(x)
        _x = Conv2D(x.shape[-1] // reduction, 1, padding="same", use_bias=True, activation="relu")(_x[:, None, None, :])
        _x = Conv2D(x.shape[-1], 1, padding="same", use_bias=True, activation="sigmoid")(_x)
        return Multiply()([x, _x])

    def RCAB(x, reduction=16):
        _x = Conv2D(x.shape[-1], 3, activation="relu", padding="same")(x)
        _x = Conv2D(x.shape[-1], 3, padding="same")(_x)
        _x = SEBlock(_x, reduction=reduction)
        return Add()([x, _x])

    def RG(x, b=20, reduction=16):
        _x = RCAB(x, reduction=reduction)
        for _ in range(b - 1):
            _x = RCAB(_x, reduction=reduction)
        _x = Conv2D(x.shape[-1], 3, padding="same")(_x)
        return Add()([x, _x])

    def RIR(x, g=10, b=20, reduction=16):
        _x = RG(x, b=b, reduction=reduction)
        for _ in range(g - 1):
            _x = RG(_x, b=b, reduction=reduction)
        _x = Conv2D(x.shape[-1], 3, padding="same")(_x)
        return Add()([x, _x])

    i = Input(inputShape)
    x = Conv2D(64, 3, padding="same")(i)
    x = RIR(x, g=g, b=b, reduction=16)
    x = Conv2D(3 * (scale ** 2), 3, padding="same")(x)
    o = Lambda(tf.nn.depth_to_space, arguments={"block_size": scale}, name="PixelShuffle")(x)

    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = RCNN()

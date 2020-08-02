import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects


def DRN(inputShape=(128, 128, 3), scale=2, b=30, f=32):
    def downSample(x):
        x = Conv2D()

    i = Input(inputShape)
    model = Model(i, o)
    model.summary()


if __name__ == "__main__":
    model = DRN()

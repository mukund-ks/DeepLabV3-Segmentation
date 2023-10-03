import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    Concatenate,
    Input,
    Dropout,
)
from keras.layers import (
    AveragePooling2D,
    GlobalAveragePooling2D,
    UpSampling2D,
    Reshape,
    Dense,
)
from keras.models import Model
from keras.applications import ResNet50, ResNet101
from keras.regularizers import l2
from keras.engine.keras_tensor import KerasTensor


def squeeze_and_excite(inputs: KerasTensor, ratio: int = 8) -> KerasTensor:
    """Function to apply Squeeze & Excitation to a feature map.

    Args:
        inputs (KerasTensor): Feature Map
        ratio (int, optional): Ratio for excitation in first dense layer. Defaults to 8.

    Returns:
        KerasTensor: Re-calibrated feature map.
    """
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(
        filters // ratio,
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)
    se = Dense(
        filters,
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False,
    )(se)

    x = init * se
    return x


def ASPP(inputs: KerasTensor) -> KerasTensor:
    """Function to apply Atrous Spatial Pyramid Pooling on features from backbone.

    Args:
        inputs (KerasTensor): Features from backbone.

    Returns:
        KerasTensor: Features with better spatial context.
    """
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y1)

    # 1x1 Convolution
    y2 = Conv2D(256, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    y2 = BatchNormalization()(y2)
    # y2 = Dropout(0.5)(y2)
    y2 = Activation("relu")(y2)

    # 3x3 Convolution, Dilation Rate - 12 or 6
    y3 = Conv2D(
        256, 3, padding="same", dilation_rate=6, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    # 3x3 Convolution, Dilation Rate - 24 or 12
    y4 = Conv2D(
        256, 3, padding="same", dilation_rate=12, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    # 3x3 Convolution, Dilation Rate - 36 or 18
    y5 = Conv2D(
        256, 3, padding="same", dilation_rate=18, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    # 1x1 Convolution on the concatenated Feature Map
    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


def createModel(modelType: str, shape: tuple[int] = (256, 256, 3)) -> Model:
    """Creates a Model with the specified backbone.

    Args:
        modelType (str): Choice of backbone. ResNet50 or ResNet101.
        shape (tuple[int]): Shape of input to the model. Defaults to (256, 256, 3).

    Returns:
        Model: Your DeepLabV3+ Model.
    """
    inputs = Input(shape)  # instantiating a tensor

    encoder = (
        ResNet101(weights="imagenet", include_top=False, input_tensor=inputs)
        if modelType == "ResNet101"
        else ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    )

    image_features = encoder.get_layer(
        "conv4_block23_out" if modelType == "ResNet101" else "conv4_block6_out"
    ).output

    # High-Level Features
    x_a = ASPP(image_features)
    # Up-Sampling High-Level Features by 4
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    x_a = Dropout(0.5)(x_a)

    # Low-Level Features
    x_b = encoder.get_layer("conv2_block2_out").output

    # 1x1 Convolution on Low-Level Features
    x_b = Conv2D(
        filters=48, kernel_size=1, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    # Concatenating High-Level and Low-Level Features
    x = Concatenate()([x_a, x_b])
    x = Dropout(0.5)(x)
    # x = squeeze_and_excite(x)

    # 3x3 Convolution on Concatenated Map
    x = Conv2D(
        filters=256, kernel_size=3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = squeeze_and_excite(x)

    # 3x3 Convolution on Concatenated Map
    x = Conv2D(
        filters=256, kernel_size=3, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Activation("relu")(x)
    x = squeeze_and_excite(x)

    # Final Up-Sampling by 4
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs, x)

    return model


if __name__ == "__main__":
    model = createModel("ResNet50")
    model.summary()

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
from keras.applications import ResNet50, ResNet101, Xception, EfficientNetB5
from keras.regularizers import l2
from tensorflow.python.keras.engine.keras_tensor import KerasTensor


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
    y1 = squeeze_and_excite(y1)

    # 1x1 Convolution
    y2 = Conv2D(256, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(inputs)
    y2 = BatchNormalization()(y2)
    # y2 = Dropout(0.5)(y2)
    y2 = Activation("relu")(y2)
    y2 = squeeze_and_excite(y2)

    # 3x3 Convolution, Dilation Rate - 6
    y3 = Conv2D(
        256, 3, padding="same", dilation_rate=6, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)
    y3 = squeeze_and_excite(y3)

    # 3x3 Convolution, Dilation Rate - 12
    y4 = Conv2D(
        256, 3, padding="same", dilation_rate=12, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)
    y4 = squeeze_and_excite(y4)

    # 3x3 Convolution, Dilation Rate - 18
    y5 = Conv2D(
        256, 3, padding="same", dilation_rate=18, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)
    y5 = squeeze_and_excite(y5)

    # 3x3 Convolution, Dilation Rate - 24
    y6 = Conv2D(
        256, 3, padding="same", dilation_rate=24, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y6 = BatchNormalization()(y6)
    y6 = Activation("relu")(y6)
    y6 = squeeze_and_excite(y6)

    # 3x3 Convolution, Dilation Rate - 36
    y7 = Conv2D(
        256, 3, padding="same", dilation_rate=36, use_bias=False, kernel_initializer="he_normal"
    )(inputs)
    y7 = BatchNormalization()(y7)
    y7 = Activation("relu")(y7)
    y7 = squeeze_and_excite(y7)

    # 1x1 Convolution on the concatenated Feature Map
    y = Concatenate()([y1, y2, y3, y4, y5, y6, y7])
    y = Conv2D(256, 1, padding="same", use_bias=False, kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = squeeze_and_excite(y)

    return y


def createModel(modelType: str, shape: tuple[int] = (256, 256, 3)) -> Model:
    """Creates a Model with the specified backbone.

    Args:
        modelType (str): Choice of backbone. ResNet50, ResNet101, Xception or EfficientNetB5.
        shape (tuple[int]): Shape of input to the model. Defaults to (256, 256, 3).

    Returns:
        Model: Your DeepLabV3+ Model.
    """
    inputs = Input(shape)  # instantiating a tensor

    def get_xception():
        return Xception(weights="imagenet", include_top=False, input_tensor=inputs)

    def get_efficientb5():
        return EfficientNetB5(weights="imagenet", include_top=False, input_tensor=inputs)

    def get_resnet50():
        return ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)

    def get_resnet101():
        return ResNet101(weights="imagenet", include_top=False, input_tensor=inputs)

    model_mappings = {
        "Xception": {
            "base_model_func": get_xception,
            "block_name": "block13_sepconv2_act",
            "low_level_name": "block4_sepconv1_act",
        },
        "EfficientNetB5": {
            "base_model_func": get_efficientb5,
            "block_name": "block7c_activation",
            "low_level_name": "block2e_activation",
        },
        "ResNet101": {
            "base_model_func": get_resnet101,
            "block_name": "conv4_block23_out",
            "low_level_name": "conv2_block2_out",
        },
        "ResNet50": {
            "base_model_func": get_resnet50,
            "block_name": "conv4_block6_out",
            "low_level_name": "conv2_block2_out",
        },
    }

    model_params = model_mappings.get(modelType, {})
    base_model_func = model_params.get("base_model_func")
    base_model = base_model_func()
    block_name = model_params.get("block_name")
    low_level_name = model_params.get("low_level_name")

    image_features = base_model.get_layer(block_name).output

    if modelType == "Xception":
        image_features = Conv2D(
            1024, (1, 1), padding="same", kernel_initializer="he_normal", use_bias=False
        )(image_features)
        image_features = BatchNormalization()(image_features)
        image_features = Activation("relu")(image_features)

    if modelType == "EfficientNetB5":
        image_features = UpSampling2D((2, 2), interpolation="bilinear")(image_features)
        image_features = Conv2D(
            1024, (1, 1), padding="same", kernel_initializer="he_normal", use_bias=False
        )(image_features)
        image_features = BatchNormalization()(image_features)
        image_features = Activation("relu")(image_features)

    # High-Level Features
    x_a = ASPP(image_features)

    # Up-Sampling High-Level Features by 4
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)
    # x_a = Dropout(0.5)(x_a)

    # Low-Level Features
    x_b = base_model.get_layer(low_level_name).output
    if modelType == "Xception":
        x_b = UpSampling2D((2, 2), interpolation="bilinear")(x_b)
    if modelType == "EfficientNetB5":
        x_b = Conv2D(256, (1, 1), padding="same", kernel_initializer="he_normal", use_bias=False)(
            x_b
        )

    # 1x1 Convolution on Low-Level Features
    x_b = Conv2D(
        filters=48, kernel_size=1, padding="same", use_bias=False, kernel_initializer="he_normal"
    )(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)
    x_b = squeeze_and_excite(x_b)

    # Concatenating High-Level and Low-Level Features
    x = Concatenate()([x_a, x_b])
    # x = Dropout(0.5)(x)

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
    model = createModel("EfficientNetB5")
    model.summary()

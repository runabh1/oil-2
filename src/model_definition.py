# src/model_definition.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, Dropout

# IMPORTANT: These functions must match the structure trained in Colab (Cell 3)

def conv_block(input_tensor, num_filters):
    """Standard Convolution Block (Conv -> BatchNorm -> ReLU) x 2"""
    # First Conv
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second Conv
    x = Conv2D(num_filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def build_unet(input_shape, num_classes):
    """Builds the U-Net model using the Keras Functional API."""
    inputs = Input(shape=input_shape)
    
    # --- Encoder (Contracting Path) ---
    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1); p1 = Dropout(0.1)(p1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2); p2 = Dropout(0.1)(p2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3); p3 = Dropout(0.2)(p3)

    # --- Bottleneck ---
    bottleneck = conv_block(p3, 512)
    bottleneck = Dropout(0.3)(bottleneck)

    # --- Decoder (Expansive Path) ---
    u4 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u4 = concatenate([u4, c3], axis=-1); u4 = Dropout(0.2)(u4)
    c4 = conv_block(u4, 256)

    u5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c2], axis=-1); u5 = Dropout(0.1)(u5)
    c5 = conv_block(u5, 128)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c1], axis=-1); u6 = Dropout(0.1)(u6)
    c6 = conv_block(u6, 64)

    # --- Output Layer ---
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c6)

    model = Model(inputs=[inputs], outputs=[outputs], name='InterpAI_UNet')
    return model

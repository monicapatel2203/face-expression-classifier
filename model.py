import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Accuracy, AUC, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def build_model2(img_shape, num_classes, fine_tune = 0, drop_out = 0):
    #     input_layer = Input(shape=img_shape)

    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape= (*img_shape, 3),
    )
    # Freeze all the layers

    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze certain layers

    if fine_tune != 0:
        for layer in base_model.layers[fine_tune:]:
            layer.trainable = True

    base_model_output = base_model.output

    x = GlobalAveragePooling2D()(base_model_output)

    x = Dense(1024, activation='relu')(x)

    if drop_out != 0:
        x = Dropout(drop_out)(x)

    output = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)
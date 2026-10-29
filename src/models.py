
import tensorflow as tf
from tensorflow.keras import layers, models


def build_mfcc_cnn(input_shape=(40, 130, 1), num_classes=8):
    """Build a CNN classifier for MFCC features."""

    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    x = layers.Conv2D(
        32, kernel_size=3, padding="same", use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(
        64, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(
        128, kernel_size=3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="emotion_probabilities",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="mfcc_emotion_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    print("Building MFCC CNN model...", flush=True)

    model = build_mfcc_cnn()
    model.summary()

    dummy_input = tf.zeros((2, 40, 130, 1))
    predictions = model(dummy_input, training=False)

    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", predictions.shape)
    print("Output classes:", predictions.shape[-1])
    print("Model architecture test completed.")

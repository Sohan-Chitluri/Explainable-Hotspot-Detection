"""Custom CNN architecture from LHD_CustomModel.ipynb (Approach 2)."""

from tensorflow.keras import layers, models


def build_custom_cnn(input_shape=(224, 224, 3)):
    """Reproduce the notebook Sequential custom CNN exactly."""
    model = models.Sequential()

    # 1st basic block
    model.add(layers.Conv2D(12, (3, 3), activation="elu", input_shape=input_shape))
    model.add(layers.Conv2D(12, (3, 3), activation="elu"))
    model.add(layers.Conv2D(12, (3, 3), activation=None))
    model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(layers.Activation("elu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Max pooling layer between basic blocks
    model.add(layers.MaxPooling2D((5, 5)))

    # 2nd basic block
    # Note: notebook repeats input_shape= on this Conv2D; Keras ignores it after the first layer.
    model.add(layers.Conv2D(12, (3, 3), activation="elu", input_shape=input_shape))
    model.add(layers.Conv2D(12, (3, 3), activation="elu"))
    model.add(layers.Conv2D(12, (3, 3), activation=None))
    model.add(layers.BatchNormalization(momentum=0.99, epsilon=0.001))
    model.add(layers.Activation("elu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dropout
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))

    # Fully connected ANN
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    return model

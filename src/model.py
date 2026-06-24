import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def build_shelf_life_model(input_shape=(224,224,3)):
    """
    builds a mobileNetV2 backbone with a custom regression head to predict a continous float value .
    """
    # 1. base feature extractor ->pretrained on imagenet
    base_model=MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # freeze the base weights so we preserve the pre trained edge detection features
    base_model.trainable=False

    # 2. input layer 
    inputs = layers.Input(shape=input_shape, name="input_image")

    # 3. native data augmentation graph (executes on gpu during training)
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.2)(x)
    x = layers.RandomBrightness(0.1)(x)

    # 4. MobileNetV2 expected preprocessing (scales pixels values between -1,1)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 5. extract features (traing=false keeps batch normalization layers forzen)
    x = base_model(x, training=False)

    # 6. regression head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)

    #output: single node with linear activation function to predict the days
    outputs=layers.Dense(1, activation='linear', name="predicted_days")(x)

    # construct model instance
    model=models.Model(inputs, outputs, name="Produce_Shelf_Life_Regressor")
    return model

if __name__ == "__main__":
    # Internal validation block to ensure the graph compiles cleanly
    model = build_shelf_life_model()
    model.summary()
    print("\n[SUCCESS] model.py compiled and validated structure cleanly.")
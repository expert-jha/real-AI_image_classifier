import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetv2
from tensorflow.keras.layers import Dense , Flatten , Dropout

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

DATASET_DIR = "Data"

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split = 0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size = (224,224),
    batch_size = 32,
    class_mode = "binary",
    subset="training"

)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(224,224),
    batch_size = 32,
    class_mode = "binary",
    subset="validation"

)

base_model = MobileNetv2(
    weights = "imagenet",
    include_top = False,
    input_shape=(224,224,3)
)

base_model.trainable=False

x = Flatten()(base_model.output)
x = Dense(128,activation='relu')(x)
x= Dropout(0.5)(x)
x = Dense(1,activation='sigmoid')(x)

model = Model(inputs = base_model.input,output=x)
model.compile(loss="binary_crossentropy",optimizer = Adam(learning_rate=0.0001), metrics=['accuracy'])

model.fit(train_data, validation_data = val_data,epochs=50)


model.save("model_name")
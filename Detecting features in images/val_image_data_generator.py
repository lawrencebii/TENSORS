from keras_preprocessing.image import ImageDataGenerator
from add_validation import  validation_dir
validation_datagen=ImageDataGenerator(rescale=1/255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300),
    class_mode='binary'
)
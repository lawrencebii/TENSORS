from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255

training_dir='horse-or-human/training/'
training_datagen = ImageDataGenerator(rescale=1/255)
train_generator = training_datagen.flow_from_directory(
    training_dir,
    target_size=(300,300),
    class_mode='binary'

)
'''
# 1. Define the training_dir
# 2. Create an instance of ImageDataGenerator called train_datagen
# 3. Specify that this will generate images for training from a
directory
The directory is training_dir
4 Indicate some hyper parameters about the data such as the target size
class_mode is usually binary when there are two types of data in this case
or categorical if more than two



'''
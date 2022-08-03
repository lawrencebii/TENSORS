import  tensorflow as tf
from keras.optimizers import RMSprop
from val_image_data_generator import validation_generator
from img_data_generator import  train_generator
#CNN ARCHITECTURE FOR HORSES OR HUMAN

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)
'''
#  THE MODEL
1 defining 16 filters each 3x3 but the input shape is 300x300 3because of the 3 color channels
There's only one neuron in the ouput layer-> we're using a binary classifier and we can get
a binary classification with just
a single neuron if we activate it with a sigmoid function
the purporse of the sigmoid is to drive the set of values toward 0 and the other toward 1
which is perfect binary classification
2. STACKING OF SEVERAL MORE CONVOLUTIONAL LAYERS
this is because our image source is quite large, and we want over time to have many smaller images each features 
highlighted.
model.summary()
shows that after all the convolutional and pooling layer the data ends up as 7x7 items
These will be activated maps that are relatively simple,containing just 49 pixels
These feature maps can be passed to the dense neural network to match the appropriate labels
with these we're going to learn 1.7M params
# TRAINING
to train the network we'll compile it with a loss function and an optimizer
In this case rge loss function can be cross entropy loss function
because there are only two classes to be classified
and a new optimizer root mean square propagation(RMSprop) that takes
the learning rate(lr) parameter that allows us to tweak the learning
we use the fit generator
'''


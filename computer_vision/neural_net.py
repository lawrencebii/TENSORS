import tensorflow as tf

'''
Checking for overfitting of the model
'''


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > .95):
            print('\nReached 95% accuracy cancelling training')
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels),(test_images,test_labels)=mnist.load_data()
# Normalizing the images such that they will be between o0 and 1
training_images = training_images/255.0
test_images = test_images/255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=50,callbacks=[callbacks])


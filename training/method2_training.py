#Use this method for training when dataset is not large enough, also data augmentation can be performed here to learn more about data.

from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

vgg16_model = VGG16(include_top = False , weights = 'imagenet' , input_shape=(150,150,3))

end_to_end = models.Sequential()
end_to_end.add(layers.Input(shape=(150,150,3)))
end_to_end.add(vgg16_model)
end_to_end.add(layers.Flatten())
end_to_end.add(layers.Dense(256 , activation='relu' , input_shape=(8192,)))
end_to_end.add(layers.Dropout(0.5))
end_to_end.add(layers.Dense(10 , activation='softmax'))

vgg16_model.trainable = False

end_to_end.compile(loss = 'categorical_crossentropy' , optimizer = 'rmsprop' , metrics = ['acc'])
end_to_end.summary()

training_generator = ImageDataGenerator(rescale=1./255 , rotation_range = 40, width_shift_range = 0.3, height_shift_range = 0.3, zoom_range = 0.4, horizontal_flip= True, fill_mode = 'nearest')
validation_generator = ImageDataGenerator(rescale=1./255)

train_generate = training_generator.flow_from_directory(train_dir , target_size=(150,150) , class_mode='categorical' , batch_size=20) #make train_dir containing training samples (2000 samples)
validation_generate = validation_generator.flow_from_directory(validation_dir , target_size=(150,150) , class_mode='categorical' , batch_size=20) #make validation_dir containing validation samples (1000 samples)

history = end_to_end.fit(train_generate , steps_per_epoch = 100, epochs=30 , validation_data=validation_generate , validation_steps=50)


loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1 , len(acc) + 1)

plt.plot(epochs , loss , 'ro', label='training_loss')
plt.plot(epochs , val_loss , 'b' , label='validation_loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs , acc, 'ro' , label='training_accuracy')
plt.plot(epochs , val_acc, 'b' , label= 'validation_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
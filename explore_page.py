from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array

img_width, img_height = 224, 224

#training
def modeltrain():
    train_data_dir = r"H:\Suri_Amrita\AI_Amrita\hackathon\train"

    validation_data_dir = r"H:\Suri_Amrita\AI_Amrita\hackathon\test"
    nb_train_samples = 47
    nb_validation_samples = 10
    epochs = 100
    batch_size = 16
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)


    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
 
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

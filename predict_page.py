import streamlit as st
from PIL import Image
import keras_applications
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array

img_width, img_height = 224, 224

#training
def modeltrain():
    train_data_dir = r"hackathon/train"

    validation_data_dir = r"hackathon/test"
    nb_train_samples = 47
    nb_validation_samples = 10
    epochs = 2
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
    
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,
                                   horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(train_data_dir,target_size=(img_width, img_height),
                                                    batch_size=batch_size,class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size=(img_width, img_height),
                                                        batch_size=batch_size,class_mode='binary')
    model.fit(train_generator,steps_per_epoch=nb_train_samples // batch_size,epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size)
    model.save_weights('model_saved.h5')
    
def load_image(image):
    image = Image.open(image)
    kkpp = image.save("dolls.png")
    return image


def main():
    modeltrain()
    st.title("File Upload Tutorial")

    menu = ["Image"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Image":
        st.subheader("Image")

    image = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])
    if image is not None:

        # To See details
        file_details = {"filename": image.name, "filetype": image.type,
                        "filesize": image.size}
        st.write(file_details)

# To View Uploaded Image
        st.image(load_image(image), width=224)

        model = load_model('model_saved.h5')

        img = load_img(
            r"C:\Users\SUDHA\Downloads\hackahon\dolls.png", target_size=(224, 224))
        img = np.array(img)

        img = img/255
        img = img.reshape(-1, 224, 224, 3)
        label = (model.predict(img) < 0.4).astype(np.int32)

        st.write(
            "Predicted Class (0 - Non-dyslexia , 1- Dyslexia): ", label[0][0])


main()

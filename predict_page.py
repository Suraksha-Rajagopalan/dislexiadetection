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
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image
import sys
import time
from textblob import Word
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

subscription_key = "1780f5636509411da43040b70b5d2e22"
endpoint = "https://prana-------------v.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
def load_image(image):
    image = Image.open(image)
    kkpp = image.save("dolls.png")
    return image


def main():
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

        img = load_img(r"dolls.png", target_size=(224, 224))
        img = np.array(img)

        img = img/255
        img = img.reshape(-1, 224, 224, 3)
        label = (model.predict(img) < 0.4).astype(np.int32)

        # st.write(
        #     "Predicted Class (0 - Non-dyslexia , 1- Dyslexia): ", label[0][0])
        if label[0][0] == 1:
            st.write("===== Read File - local =====")
# Get image path
            read_image_path = os.path.join (r"C:\Users\cody\Desktop\studies\hackahon\hackathon\train\dyslexia", "doll.jpeg")
# Open the image
            read_image = open("dolls.png", "rb")

# Call API with image and raw response (allows you to get the operation location)
            read_response = computervision_client.read_in_stream(read_image, raw=True)
# Get the operation location (URL with ID as last appendage)
            read_operation_location = read_response.headers["Operation-Location"]
# Take the ID off and use to get results
            operation_id = read_operation_location.split("/")[-1]

# Call the "GET" API and wait for the retrieval of the results
            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status.lower () not in ['notstarted', 'running']:
                    break
                st.write('Waiting for result...')
                time.sleep(10)
            
# Print results, line by line
            text=''
            count = 0
            gra_count = 0
            count_words = 0
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        st.write(line.text)
                        count_words+=1
                        text = text + line.text
            #print(line.bounding_box)
                        word = Word(line.text)
                        matches = tool.check(line.text)
                        gra_count = len(matches)
                        gra_count+=gra_count
                        result = word.correct()
                        if result!=word:
                            count+=1
            print()
            st.write("The number of Spelling Mistakes Made are: ",count)
            st.write("The number of Grammatical Mistakes Made are: ",gra_count)
            st.write("The number of gramatic error of whole text is: ",len(tool.check(text)))
            st.write("End of Computer Vision quickstart.")

            if (count>(count_words*0.05) or gra_count>count):
                st.write("The person MAY suffer dyslexia")
            elif count>(count_words*0.1):
                st.write("This person IS dyslexic")
            else:
                st.write("This person MAY NOT have dyslexia")


main()

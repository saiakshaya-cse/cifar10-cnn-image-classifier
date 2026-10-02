import tensorflow as tf
from PIL import Image
import numpy as np

#cifar10 class namees
class_names=["airplane",
             "automobile",
             "bird",
             "cat",
             "deer",
             "dog",
             "frog",
             "horse",
             "ship",
             "truck"
             ]

#Load the model
model=tf.keras.models.load_model("best_model.keras")

#Load the image the Random image to test the model
image=Image.open("images/test_image.jpg").convert("RGB")

#Resize the image to 32x32
image=image.resize((32,32))

#Convert the image to numpy array
image_array=np.array(image)

#Normalize the image
image_array=image_array/255.0

#Add batch dimension
image_array=np.expand_dims(image_array,axis=0)

#Make prediction
prediction=model.predict(image_array)

#Find the class with highest probability
predicted_class=tf.argmax(prediction[0]).numpy()

#Get the confidence 
confidence=prediction[0][predicted_class]*100

#Results
print("predicted class:",class_names[predicted_class])
print("confidence:",round(confidence,2),"%")

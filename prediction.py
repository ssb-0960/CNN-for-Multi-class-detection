import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

loaded_model = tf.keras.models.load_model(r'C:\Users\hehe0\PycharmProjects\cnn\avmodel_multi_class.h5')

image_path = r'D:\dataset\crop_images\testing data\rust\rust_leaf_1295.jpg'

img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

predictions = loaded_model.predict(img)
predictions = np.squeeze(predictions)

predicted_class_index = int(np.argmax(predictions))

class_names = ['Healthy', 'Rot', 'Rust']

if 0 <= predicted_class_index < len(class_names):
    predicted_class_name = class_names[predicted_class_index]
else:
    predicted_class_name = "Unknown"

print(f'The predicted class is: {predicted_class_name}')

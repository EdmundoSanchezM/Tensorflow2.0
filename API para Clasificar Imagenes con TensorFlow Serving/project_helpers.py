#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 19:43:12 2019

@author: JuanGabriel
"""


from tensorflow.keras.datasets import fashion_mnist
import imageio

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

for i in range(5):
    imageio.imwrite(uri="uploads/{}.png".format(i), im=X_test[i])


#Fransform images

from PIL import Image

def resize_and_convert_to_grayscale(image_path, target_size=(32, 32),color=False):
    try:
        # Open the image file
        image = Image.open(image_path)

        # Resize the image to the target size (28x28)
        image = image.resize(target_size, Image.ANTIALIAS)

        if(not color):
            # Convert the image to grayscale (L mode)
            image = image.convert("L")

        # Save the resized and grayscale image
        save_path = image_path.replace("caballo.jpg", "_resized.png")  # Modify the filename as needed
        image.save(save_path)

        return save_path

    except Exception as e:
        print("Error:", str(e))
        return None

input_image_path = "uploads/caballo.jpg"
output_image_path = resize_and_convert_to_grayscale(input_image_path,color=True)

if output_image_path:
    print(f"Image resized and converted. Saved as {output_image_path}")
else:
    print("Something went wrong.")



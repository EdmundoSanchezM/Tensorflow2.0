#Paso 1: Importar todas las dependencias del proyecto
import os
import requests
import numpy as np
import tensorflow as tf

#Hacer el downgrade a la versión 1.1.0 con pip install scipy==1.1.0
import imageio
from flask import Flask, request, jsonify

print(tf.__version__)

#Paso 2: Cargar el modelo pre entrenado
model = tf.keras.models.load_model("modelo_cnn_FASHION.h5")

# Verifica la arquitectura del modelo
model.summary()

#Paso 3: Crear la API con Flask
#Crear una aplicación de Flask
app = Flask(__name__)

#Definir la función de clasificación de imágenes
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #Definir donde se encuentra la carpeta de imágenes
    upload_dir = "uploads/"
    #Cargar una de las imágenes de la carpeta
    image = imageio.imread(upload_dir + img_name)
    
    #Definir la lista de posibles clases de la imagen
    classes = ["Camiseta", "Pantalón", "Sudadera", "Vestido", "Abrigo", 
               "Sandalia", "Jersey", "Zapatilla", "Bolsa", "Botas"]

    #Hacer la predicción utilizando el modelo pre entrenado
    prediction = model.predict([image.reshape(1, 28*28)])

    #Devolver la predicción al usuario
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#Iniciar la aplicación de Flask
app.run(port=5000, debug=False)
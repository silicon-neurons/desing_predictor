# desing_predictor
Red neuronal convolucional para clasificación de imágenes con TensorFlow y Keras

## Construcción del modelo

Se construyó un modelo con 4 capas ocultas, que son 2 capas de convolución y 2 capas de agrupación.
La primera capa de convolución recibe imágenes de 64 x 64 x 3 de entrada y usando relu para la activación.
```
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
```
La segunda capa de convolución recibe las imágenes provenientes de la primera capa, con dimensiones de 32 x 3 x 3.
```
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
```
Las capas de agrupación usan un desfase de 2 x 2.
```
classifier.add(MaxPooling2D(pool_size = (2, 2)))
```
Finalmente, después de varias capas convolucinales y de agrupación, el razonamiento de alto nivel en la red neuronal se realiza a través de capas totalmente conectadas.

```
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 3, activation = 'softmax'))
```
## Definir el tipo de modelo
Se desarrolló un modelo con categorías, en el cual se consta de 3 categorías: Nudge, Persuasive y Unpleasant.
```
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
```
## Entrenar el modelo
Se definen los set de imágenes para entrenamiento y para prueba. Definiendo el tamaño de las imágenes como se lo definimos al modelo y se seleccionó el modo, que en este caso es por categorías.
```
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')
```
Se realizó el entrenamiento con 10 pasadas
```
classifier.fit_generator(training_set,
                         samples_per_epoch = 4000,
                         nb_epoch = 10,
                         validation_data = test_set,
                         nb_val_samples = 1000,
                        callbacks = [checkpointer])
```

## Probar el modelo
Ya teniendo el modelo entrenado, para realizar pruebas de la presición de la clasificación de imágenes, solo se necesita una imagen para probar.
```
test_image = image.load_img('random.png', target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
if result[0][0] >= 0.5:
    prediction = 'Nudge'
elif result[0][1] >= 0.5:
    prediction = 'Persuasive'
else:
    prediction = 'Unpleasant' 
print(prediction)
```
## Guardar el modelo
Para poder usar el modelo desde un webservice, se necesita exportarlo. Primero se guarda la estructura del modelo en un archivo de json y posteriormente se guardan los pesos en un archivo h5.
```
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")
print("Saved model to disk")
```

## Cargar el modelo
Para importa el modelo, solo debemos cargar el modelo y posteriormente cargar sus respectivos pesos.
```
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("model.h5")
print("Loaded model from disk")
```

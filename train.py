import os,sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file
import matplotlib.pyplot as plt
import gdown
CLASSES = ["Biodegradable", "NonBiodegradable", "No Object Found"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
IN_COLAB = 'google.colab' in sys.modules
DATASET = ""
train_gen = None
val_gen = None
model=None
def loadDataForTraining(dataset):
  global DATASET
  print(f"Loading data from {dataset}...")
  workingdir = os.getcwd()
  fileName = "waste_dataset.zip"
  gdown.download(id=dataset, output='waste_dataset.zip', quiet=False)
  fileLink = 'file://'+os.path.join(workingdir,fileName)
  file_path = get_file(fname=fileName, origin=fileLink, extract=True)
  keras_datasets_dir = os.path.dirname(file_path)
  DATASET = os.path.join(keras_datasets_dir, "dataset")
  print("Data Loaded")
def processDataForTraining():
  global train_gen, val_gen
  datagen = ImageDataGenerator(
      rescale=1.0 / 255,
      rotation_range=20,
      zoom_range=0.15,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.15,
      horizontal_flip=True,
      fill_mode="nearest",
      validation_split=0.2,
  )
  train_gen = datagen.flow_from_directory(
      DATASET,
      target_size=IMAGE_SIZE,
      class_mode="categorical",
      classes=CLASSES,
      batch_size=BATCH_SIZE,
      subset="training",
  )
  val_gen = datagen.flow_from_directory(
      DATASET,
      target_size=IMAGE_SIZE,
      class_mode="categorical",
      classes=CLASSES,
      batch_size=BATCH_SIZE,
      subset="validation",
  )
def trainTheModel():
  global model
  base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(len(CLASSES), activation="softmax")(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  for layer in base_model.layers:
      layer.trainable = False
  model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
  model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)
def evaluateModel():
  print("Evalutaing the model...")
  plt.figure(figsize=(7,5))
  plt.title("RMSE loss over epochs",fontsize=16)
  plt.plot(np.sqrt(model.history.history['loss']),c='k',lw=2)
  plt.grid(True)
  plt.xlabel("Epochs",fontsize=14)
  plt.ylabel("Root-mean-squared error",fontsize=14)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.show()
def saveModel():
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.experimental_new_converter=True
  converter._experimental_lower_tensor_list_ops = False
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,\
                                          tf.lite.OpsSet.SELECT_TF_OPS]
  tflite_model = converter.convert()
  # Save the model.
  with open('wastemanagement.tflite', 'wb') as f:
    f.write(tflite_model)
  print("Saved")
  if IN_COLAB:
    from google.colab import files
    files.download('wastemanagement.tflite')
loadDataForTraining('176U1b13disRVlVZyA3tZ4_BhV7hBvE0P')
processDataForTraining()
trainTheModel()
evaluateModel()
exit()
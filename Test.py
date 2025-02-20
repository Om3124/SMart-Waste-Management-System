cam_port = None
import sys, os
HOME        = os.path.expanduser('~')
RPI_HOME    = HOME + '/RPI/'
GROK_HOME   = HOME + '/Desktop/Grok-Downloads/'
sys.path.insert(1, RPI_HOME)
from file_watcher import FileWatcher, device_sensor
from grok_library import check_with_simulator,check_with_simulator2, device, sim_device, pin, GrokLib
import threading
grokLib = GrokLib()
device['applicationIdentifier'] = str(os.path.splitext(os.path.basename(__file__))[0])
device['mobile_messages'] = list()
def simulate(list_of_sensors):
    if list_of_sensors is not None:
        global sim_device
        sim_device = list_of_sensors
def startListener1():
    FileWatcher(simulate, 'simulation.json', RPI_HOME, 'config_file')
thread1 = threading.Thread(target=startListener1, args=())
thread1.daemon=True
thread1.start()
from cv2 import *
cam_port = 0
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
result = None
def getPredictionsFromModel():
    model_path = "/home/pi/Desktop/Grok-Downloads/wastemanagement.tflite"
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    categories = ["Biodegradable", "NonBiodegradable", "No Object Found"]
    input_details = interpreter.get_input_details()
    dtype =  input_details[0]['dtype']
    Shape = input_details[0]['shape']
    frame = cv2.imread("/home/pi/Desktop/Grok-Downloads/image.jpg")
    image = cv2.resize(frame, (224, 224))
    image = image.reshape(Shape)
    image = image.astype(dtype)
    image = image / 255.0
    input_tensor_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_tensor_index, image)
    interpreter.invoke()
    output_tensor_index = interpreter.get_output_details()[0]['index']
    prediction = interpreter.get_tensor(output_tensor_index)[0]
    class_idx = np.argmax(prediction)
    class_label = categories[class_idx]
    confidence = prediction[class_idx] * 100
    return class_label, confidence
def getResults():
    class_label, confidence = getPredictionsFromModel()
    print(f"Class: {class_label}, Confidence: {confidence:.2f}%")
    return class_label, confidence
while True:
  cam_port = 0
  cam = VideoCapture(cam_port)
  result, image = cam.read()
  if result:
  	imwrite("/home/pi/Desktop/Grok-Downloads/image.jpg", image)
  cam.release()
  getPredictionsFromModel()
  device["mobile_messages"].append({'type' : 'text', 'value' : ('Result: ' + str(getResults())), 'color' : '#33cc00'})
  image_url = grokLib.upload_image('/home/pi/Desktop/Grok-Downloads/image.jpg')
  device['mobile_messages'].append({'type' : 'image','source' : image_url,'state' : True})
  device_sensor(device)
  device["mobile_messages"] = []

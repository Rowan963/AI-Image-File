import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import glob

def aidemy_imshow(name,img):
    b,g,r=cv2.split(img)
    img=cv2.merge([r,g,b])
    plt.title(name)
    plt.imshow(img)
    plt.show()
    
cv2.imshow=aidemy_imshow

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 10
FONT_THICKNESS = 10
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  list1=[]
  
  for detection in detection_result.detections:
    
    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    list1.append(result_text)
  return list1

for i in glob.iglob('*.PNG'):
    IMAGE_FILE = os.path.abspath(i)
    
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print("Failed to load image.")
    
    dir1='C:\Python learning\Image Detection (project)\imgs'
    
    base_options = python.BaseOptions(model_asset_path=r"C:\Python learning\Image Detection (project)\efficientdet.tflite")
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.2)
    
    detector = vision.ObjectDetector.create_from_options(options)
    image = mp.Image.create_from_file(IMAGE_FILE)
    detection_result = detector.detect(image)
    image_copy = np.copy(image.numpy_view())
    
    annotated_image = visualize(image_copy, detection_result)
    os.rename(IMAGE_FILE,dir1+list1)





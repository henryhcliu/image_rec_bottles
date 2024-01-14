from imageai.Detection import ObjectDetection
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
execution_path = os.getcwd()
 
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
# select an appropriate model path for a specific task
detector.setModelPath( os.path.join(execution_path , "yolov3.pt"))
detector.loadModel()
custom_objects = detector.CustomObjects(bottle=True)
# detect from camera
camera = cv2.VideoCapture(0)
while True:
    return_value,image = camera.read()
    # Reduce resolution to 240P
    image = cv2.resize(image,(640,480))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('original image',image)
    # detect every frame
    returned_img, detections = detector.detectObjectsFromImage(input_image=image, output_type="array", minimum_percentage_probability=20)
    # calculate the number of bottles
    bottle_count = 0
    for eachObject in detections:
        if eachObject["name"] == "bottle":
            bottle_count += 1
    print("Number of bottles: ", bottle_count)
    # show the number of bottles on the image
    cv2.putText(returned_img, "Number of bottles: "+str(bottle_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Plastic Bottle Detection for the BottleUp Project',returned_img)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
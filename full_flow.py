from detect import *

source = "./data/images/image000009.jpg"
inputImage = cv2.imread(source)


def vehicle_detected(imageInput):
    image1, boxes_detected = vehicle_detected_model(image=imageInput)
    return image1, boxes_detected


image, boxes = vehicle_detected(inputImage)
print(boxes)
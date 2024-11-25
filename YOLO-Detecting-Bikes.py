
from ultralytics import YOLO
import cv2
import cvzone
import math

# For vidoes
cap = cv2.VideoCapture("traffic.mp4")
# we-cam object
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)  # height
# cap.set(4,720)   # width

# model creation
model = YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic Light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat","Cell phone",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
"dining table", "toilet", "tv monitor", "Laptop", "mouse", "remote", "keyboard", "cell phone",
"micro mave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush","Safety helmet"]

# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#             w, h = x2 - x1, y2 - y1
#             # Corrected line: pass the image and the bounding box separately
#             img = cvzone.cornerRect(img, (x1, y1, w, h))
#
#             # confidence
#             conf = math.ceil((box.conf[0]*100))/100
#             print(conf)
#             cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(0,y1-20)))
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(1)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1


            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            # cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            # Classification
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'motorbike' or currentClass == 'Safety helmet' and conf > 0.3:
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),scale=0.6,thickness=0,offset=3)
                cvzone.cornerRect(img, (x1, y1, w, h),l=9)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
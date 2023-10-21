from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# cap = cv2.VideoCapture(0) # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("./Videos/ppe-2.mp4")  # For Video


model = YOLO("best_ppe.pt")

classNames = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

my_color = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))            

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]
            if conf > 0.5:
                if current_class == "Hardhat" or current_class == "Safety Vest":
                    my_color = (0, 255, 0)
                elif current_class == "Person":
                    continue
                else:
                    my_color = (0, 0, 255)

                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}",
                    (max(0, x1), max(35, y1)),
                    scale=0.8,
                    thickness=2,
                    colorB=my_color,
                    colorT=(255, 255, 255),
                    colorR=(my_color),
                    offset=10,
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

from ultralytics import YOLO
import cv2
import math
import cvzone


model=YOLO("models/best.pt")

cap=cv2.VideoCapture("videos\ppe-3-1.mp4")
class_name=['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
while True :
    suc,image=cap.read()
    result=model(image)
    for r in result:
        boxes=r.boxes
        for box in boxes:
            cls=int(box.cls[0])
            conf=math.ceil(box.conf[0]*100)/100
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(image,(x1,y1),(x2,y2),color=(255,0,255),thickness=10)
            cvzone.putTextRect(image,f"{class_name[cls]}  {conf}",(x1,y1-35),scale=1)
    cv2.imshow("Detection",image) 
    cv2.waitKey(1)
            

# result=model.predict("images/bus.jpg",show=True,imgsz=40)
# cv2.waitKey(0)


# print(result)
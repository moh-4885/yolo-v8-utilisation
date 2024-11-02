from ultralytics import YOLO
import time
import cv2
import cvzone
model =YOLO("./models/yolov8l.pt")
import math
from sort import *
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
def line_eq(x):
    return 1.875*x-448.12
cap =cv2.VideoCapture("./videos/people.mp4")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
viucle={}
mask=cv2.imread("./images/elivator_mask.png")
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
up=[]
down=[]
limits = [430, 297, 673, 297]
limits1 = [157, 460, 430, 460]
a=0
try :
    while True:
        success,image=cap.read()
        mask=cv2.resize(mask,(1280,720))
        # print(mask.shape)
        # print(image.shape)

        masked_image=cv2.bitwise_and(image,mask)
        # if a==0:
        #     cv2.imwrite("cor.png",masked_image)
        #     a=1
        result=model(masked_image,stream=True)
        detections=np.empty((0,5))
        for r in result:
            boxes=r.boxes
            for box in boxes:
                # x1,y1,x2,y2=box.xyxy[0]
                # x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                # # w,h=x2-x1,y2-y1
                # # cvzone.cornerRect(image,(x1,y1,w,h))
                conf=box.conf[0]
                conf=math.ceil(conf*100)/100
                cls=int(box.cls[0])
                if cls == 0:
                    x1,y1,x2,y2=box.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

                    detections=np.vstack((detections,np.array([x1,y1,x2,y2,conf])))
                # if classNames[cls]   not in viucle:
                #     viucle[classNames[cls]] = 0
                # viucle[classNames[cls]]=viucle[classNames[cls]]+1
                # cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,0,255))
                # cvzone.putTextRect(image,f"{classNames[cls]} number {conf}",(max(x1,0),max(y1-20,35)),scale=1)
        trackerResult=tracker.update(detections)
        for rt in trackerResult:
            x1,y1,x2,y2,id=rt
            x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
            w,h=x2-x1,y2-y1
            cx,cy=x1+(w/2),y1+(h/2)
            if cy >line_eq(cx) and cy <limits1[1]-10:
                if id not in up:
                    up.append(id)
            elif cy < line_eq(cx) and cy > limits[1]+10:
                if id not in down:
                    down.append(id)             
            
            cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,0,255))
            cvzone.putTextRect(image,f" {id}",(max(x1,0),max(y1-20,35)),scale=1)
           
               
        print(up,down)        
        cvzone.putTextRect(image,f"UPS :{len(up)}  Downs : {len(down)}",(810, 260))
        cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),color=(255,0,255),thickness=3)
        cv2.line(image,(limits1[0],limits1[1]),(limits1[2],limits1[3]),color=(255,0,255),thickness=3)
        cv2.imshow("image",image)
        cv2.waitKey(1)
except Exception as e:
    # print(len(total_ids))
    print(e)
    
    

            

            
            
            
    


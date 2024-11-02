from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *



model =YOLO("../models/yolov8l.pt")
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
cap =cv2.VideoCapture("../videos/cars.mp4")
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

print(os.path.abspath("./"))
mask=cv2.imread("../images/mask-950x480.png")
tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)
total_ids=[]
try :
    while True:
        success,image=cap.read()
        mask=cv2.resize(mask,(1280,720))
        # print(mask.shape)
        # print(image.shape)
        limits = [400, 297, 673, 297]
        masked_image=cv2.bitwise_and(image,mask)
        result=model(masked_image,stream=True)
        detections=np.empty((0,5))
        for r in result:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                # w,h=x2-x1,y2-y1
                # cvzone.cornerRect(image,(x1,y1,w,h))
                conf=box.conf[0]
                conf=math.ceil(conf*100)/100
                cls=int(box.cls[0])
                if cls in [2,3,5,7]:
                    detections=np.vstack((detections,np.array([x1,y1,x2,y2,conf])))
        trackerResult=tracker.update(detections)
        for rt in trackerResult:
            x1,y1,x2,y2,id=rt
            x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
            w,h=x2-x1,y2-y1
            cx,cy=x1+(w/2),y1+(h/2)
            cv2.circle(image,(int(cx),int(cy)),radius=5,color=(255,0,0))
            if cy >limits[1]+35 and id not in total_ids:
                 total_ids.append(id)
            
            cv2.rectangle(image,(x1,y1),(x2,y2),color=(0,0,255))
            cvzone.putTextRect(image,f" {id}",(max(x1,0),max(y1-20,35)),scale=1)
           
               
                
        cvzone.putTextRect(image,f"Total count :{len(total_ids)}",(50,50))
        cv2.line(image,(limits[0],limits[1]),(limits[2],limits[3]),color=(255,0,255),thickness=3)
        cv2.imshow("image",image)
        cv2.waitKey()
except Exception as e:
    print(len(total_ids))
    print(e)
    
    

            

            
            
            
    


import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0,0],
    [1,0],
    [1,1],
    [0,1]
])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Yolov8 Object Detection')
    parser.add_argument(
        '--webcam-resolution',
        nargs=2,type=int,
        default=[1280,720]
        )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
    )
    return parser.parse_args() 

def getPersonCount(itemList):
    personCount = 0
    for item in itemList:
        if item == 0:
            personCount += 1
    return personCount

def main():
    args = parse_args()
    
    webcam = cv2.VideoCapture(0)   
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, args.webcam_resolution[0])
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.webcam_resolution[1])
    
    model = YOLO("yolov8l.pt")
    
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=1
    )
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(zone_polygon,frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(zone,color=sv.Color.blue())
    
    while True:
        ret, frame = webcam.read()
        
        result = model(frame,device="mps")[0] 
        
        detections = sv.Detections.from_yolov8(result)
        
        labels = [
            f"{model.model.names[class_id]}: {confidince:.2f}"
            for _, confidince, class_id, _ in detections
        ]
             
        frame = box_annotator.annotate(scene=frame, detections=detections,labels=labels)
        
        zone.trigger(detections)
        frame = zone_annotator.annotate(scene=frame)
        
        cv2.putText(frame, str(getPersonCount(detections.class_id)),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
        
        cv2.imshow('yolov8', frame)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
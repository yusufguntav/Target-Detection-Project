import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time

PERSON_COUNT = 0
ZONE_POLYGON = np.array([
    [.5,1],
    [.507,1],
    [.507,0],
    [.5,0]
])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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

def increase_person_count(increase):
    global PERSON_COUNT
    if  any(increase.tolist()):
        PERSON_COUNT += 1
   
    
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
        frame = cv2.flip(frame, 1)
        result = model(frame,device="mps")[0] 
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]
            
        
        labels = []
        for _, (xyxy, confidence, class_id, _) in enumerate(detections):
            x1, y1, x2, y2 = xyxy 
            width = x2 - x1
            height = y2 - y1
            labels.append(f"{model.model.names[class_id]}: {confidence:.2f} H:{height}")

        
        frame = box_annotator.annotate(scene=frame, detections=detections,labels=labels)
           
        increase =  zone.trigger(detections)
        increase_person_count(increase)
        frame = zone_annotator.annotate(scene=frame)
        
        cv2.putText(frame, str(PERSON_COUNT),(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 
        
        cv2.imshow('Target Detection Project', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
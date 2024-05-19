import cv2
import argparse
from ultralytics import YOLO
import supervision as sv

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
    
    while True:
        ret, frame = webcam.read()
                
        result = model(frame)[0]
        detections = sv.Detections.from_yolov8(result)
        
        frame = box_annotator.annotate(scene=frame, detections=detections)
        
        cv2.imshow('yolov8', frame)

        if cv2.waitKey(1) == ord('q'):
            break

if __name__ == '__main__':
    main()
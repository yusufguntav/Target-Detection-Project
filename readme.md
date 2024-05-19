Setup
1-Create your venv, Example: python3 -m venv venv
2-source venv/bin/activate
3-pip install ultralytics
4-pip install supervision==0.2.0

Test
yolo predict source=0  model=yolov8l.pt show=true
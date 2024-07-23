#yolov8-opencv-cpp
Example of performing inference with ultralytics YOLO V8, OpenCV 4.5.4 DNN, C++ 

Example of YOLO v8 detection on video file

Prerequisites
Make sure you have already on your system:

OpenCV 4.5.4+
GCC 9.0+ (only if you are intended to run the C++ program)
IMPORTANT!!! Note that OpenCV versions prior to 4.5.4 will not work at all.

Which YOLO version should I use?
This repository uses YOLO V8 and YOLO V5 but it is not the only YOLO version out there. You can read this article to learn more about YOLO versions and choose the more suitable one for you.

Exporting yolo v8 models to .onnx format
Check here: ultralytics/yolov8

My commands were:
yolo export model=yolov8s.pt format=onnx
And then it will  convert the model in cmd.

>yolo export model=yolov8s.pt format=onnx
Downloading https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt to 'yolov8s.pt'...
100%|█████████████████████████████████████████████████████████████████████████████| 21.5M/21.5M [00:01<00:00, 16.9MB/s]
Ultralytics YOLOv8.2.58 🚀 Python-3.12.3 torch-2.3.1+cpu CPU (12th Gen Intel Core(TM) i5-1235U)
YOLOv8s summary (fused): 168 layers, 11,156,544 parameters, 0 gradients, 28.6 GFLOPs

PyTorch: starting from 'yolov8s.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 84, 8400) (21.5 MB)

ONNX: starting export with onnx 1.16.1 opset 17...
ONNX: export success ✅ 3.3s, saved as 'yolov8s.onnx' (42.8 MB)

Export complete (9.0s)
Results saved to D:\OPENCV\yolov8n
Predict:         yolo predict task=detect model=yolov8s.onnx imgsz=640
Validate:        yolo val task=detect model=yolov8s.onnx imgsz=640 data=coco.yaml
Visualize:       https://netron.app
💡 Learn more at https://docs.ultralytics.com/modes/export

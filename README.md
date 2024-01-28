# Vehicle Management System
![image](https://github.com/tmkhang1999/YOLOv8-DeepSORT-PaddleOCR-Supervision-Vehicle-Management-System/assets/74235084/9b90b252-9929-4b7c-ae64-efc33880975c)

https://github.com/tmkhang1999/YOLOv8-DeepSORT-PaddleOCR-Supervision-Vehicle-Management-System/assets/74235084/4b8cd9f3-6de2-4060-848e-a12cfafd97b8

## Table of Contents
1. [Motivation](#motivation)
2. [Installation](#installation)
3. [File Description](#files)
4. [Acknowledgements](#acknowledgements)

## Motivation<a name="motivation"></a>
In recent years, there has been a rapid growth in the field of AI. This growth has led us to explore how we can leverage advanced technology for the enhancement of highway traffic camera systems. The primary motivation behind this project is to seamlessly integrate cutting-edge deep learning algorithms into our vehicle management system. In pursuit of this goal, our project aims to accomplish the following objectives:

1. **Detect and classify vehicles** on the road, including cars, trucks, buses, and motorcycles.
2. **Count vehicles** as they enter and exit the monitored area by implementing a robust **tracking system**.
3. **Estimate the speed** on 2D frames in real-time.
4. **Detect and recognize license plates**, enhancing the system's capabilities further.

By achieving these objectives, we aspire to create a more efficient and intelligent traffic monitoring system that leverages the power of AI to enhance safety and traffic management on highways.

## Installation <a name="installation"></a>
- Python >= 3.6
- supervision ~= 0.1.0
- ultralytics ~= 8.0.190
- deep-sort-realtime == 1.3.1
- paddleocr >= 2.0.1
- PyMuPDF == 1.21.1
- numpy < 1.24
- opencv-python ~= 4.8.1.78
- pyaml_env ~= 1.2.1

To install this project and its dependencies, follow these steps:
1. Clone this repository:
```angular2html
git clone https://github.com/tmkhang1999/YOLOv8-DeepSORT-PaddleOCR-Supervision-Vehicle-Management-System.git
```
2. Install the project's dependencies with pip:
```angular2html
pip install requirements.txt
```
3. Run command:
```angular2html
python main.py --config-path ./utils/config.yml --source-video ./data/videos/vehicle-counting.mp4 --target-video ./output.mp4
```
Or you can click on Vehicle_Quick_Test notebook I provided above to quickly run the test and see the results.

## File Description <a name="files"></a>
```angular2html
├── README.md
├── main.py *** running ConfigManager and VideoProcessor
├── requirements.txt
├── data
│   ├── models
│   │   ├── yolov8x.pt *** weight for vehicle detection
│   │   ├── license_plate_detector.pt
│   ├── videos
│   │   ├── vehicle-counting.mp4 *** input video
│   │   ├── output.mp4
├── modules
│   ├── annotation.py *** annotate the notes and tracking trails
│   ├── plate_recognition.py *** detect, recognize, and annotate plates
│   ├── speed_estimation.py *** estimate speed utilizing 2D camera's info
│   ├── video_processor.py *** the core class utilizing supervision, yolov8, deepsort, paddleOCR
└── utils
    ├── config.py *** ConfigManager to load config file
    └── config.yml
```

## Acknowledgements<a name="acknowledgements"></a>
We appreciate the following repositories for their valuable contributions to our project:
- [roboflow/supervision](https://github.com/roboflow/supervision) for streamlining our code and simplifying the annotation process.
- [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) for providing YOLOv8 models and weights.
- [levan92/deep_sort_realtime](https://github.com/levan92/deep_sort_realtime) for offering an easy-to-use deepsort implementation.
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for supplying an efficient OCR model.
- [NVIDIAAICITYCHALLENGE/2018AICITY_Maryland](https://github.com/NVIDIAAICITYCHALLENGE/2018AICITY_Maryland/tree/master) for making available the code for speed estimation on 2D frames.

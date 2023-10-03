# <p style="text-align: center">Vehicle Management System</p>

Demo

## Table of Contents
1. [Motivation](##motivation)
2. [Installation](##installation)
3. [File Descriptions](##files)
4. [Acknowledgements](##acknowledgements)

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

## Motivation<a name="motivation"></a>
In recent years, there has been a rapid growth in the field of AI. This growth has led us to explore how we can leverage advanced technology for the enhancement of highway traffic camera systems. The primary motivation behind this project is to seamlessly integrate cutting-edge deep learning algorithms into our vehicle management system. In pursuit of this goal, our project aims to accomplish the following objectives:

- **Detect and classify vehicles** on the road, including cars, trucks, buses, and motorcycles.
- Implement a robust **tracking system** to accurately count vehicles as they enter and exit the monitored area.
- Utilize 2D camera data to **estimate the speed** of vehicles in real-time.
- **Detect and recognize license plates**, enhancing the system's capabilities further.

By achieving these objectives, we aspire to create a more efficient and intelligent traffic monitoring system that leverages the power of AI to enhance safety and traffic management on highways.

## File Descriptions <a name="files"></a>
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
https://github.com/roboflow/supervision

https://github.com/ultralytics/ultralytics

https://github.com/levan92/deep_sort_realtime

https://github.com/PaddlePaddle/PaddleOCR


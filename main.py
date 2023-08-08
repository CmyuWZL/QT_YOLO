# 创建者：Cmyu
# 创建时间： 2023-06-08 23:29

import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QDesktopServices
from PyQt5.QtWidgets import QApplication, QMainWindow, \
    QLabel, QVBoxLayout, QWidget, QSlider, QPushButton, QFileDialog


class Video_detect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("A Normal Camera")
        self.setGeometry(100, 100, 800, 600)  # 设置窗口的几何形状
        # self.setStyleSheet("background-color: gray")  # 设置窗口的背景颜色   #0 #EEB8B8, stop: 1 #FADCDA
        self.setStyleSheet('''
            QMainWindow {
                border: 10px solid #6ECEDA;  /* 边框颜色和宽度 */
                border-radius: 20px;  /* 圆角半径 */
                
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                    stop: 0 #C3E2DD, stop: 1 #6ECEDA);  /* 设置垂直渐变背景色 */ 
            }
        ''')
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { "
                                       "background: transparent"
                                       "}")
        self.video_text = 'close'
        self.recognition_text = 'close'
        self.yolo_recognition_text = 'close'
        self.video_stream_label = QLabel(f"Video Stream:{self.video_text}  "
                                         f"Face Detection:{self.recognition_text}  "
                                         f"YOLO :{self.yolo_recognition_text}")
        self.video_stream_label.setStyleSheet("QLabel "
                                              "{ background-color:#C9DECF;"
                                              " border-radius: 30px;"
                                              " border: 8px solid white; "
                                              "color: #83B1C9; }"
                                              )
        self.video_stream_label.setFixedHeight(100)
        # self.video_stream_label.setFixedWidth(1600)
        self.video_stream_label.setAlignment(Qt.AlignCenter)
        # self.video_stream_label.setStyleSheet("color: pink;")
        self.font = QFont("Arial", 12, QFont.Bold)  # 设置字体
        self.video_stream_label.setFont(self.font)
        self.upload = False  # 上传标志位
        self.file_path = ''  # 储存读到的文件的地址
        self.yolo_threshold = 0.5  # 设置yolo算法检测时的置信度阈值
        self.nms_threshold = 0.5  # 设置nms算法时的阈值
        self.fps_time = 30

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        #  文件上传按键
        self.upload_button = QPushButton("Upload File")
        self.upload_button.clicked.connect(self.upload_file)
        self.upload_button.setStyleSheet(
            '''
            QPushButton {
            background-color: #C9DECF;
            border-radius: 20px;
            border: 4px solid white;
            padding: 6px;
            color: #83B1C9;
        }
            QPushButton:hover {
            background-color: pink;
        }
    
        QPushButton:pressed {
            background-color: #B97687;
        }
            '''
        )
        self.upload_button.setFont(self.font)
        self.upload_button.setFixedSize(200, 50)
        layout.addWidget(self.upload_button, alignment=Qt.AlignCenter)

        #  返回电脑摄像头按钮
        self.back_button = QPushButton("back")
        self.back_button.clicked.connect(self.back_to_camera)
        self.back_button.setStyleSheet(
            '''
            QPushButton {
            background-color: white;
            border-radius: 20px;
            border: 6px solid #C8DBDC;
            padding: 6px;
            color: #83B1C9;
            
        }
            QPushButton:hover {
        background-color: gray;
        }
    
        QPushButton:pressed {
        background-color: #E098AE;
        }
            '''
        )
        self.back_button.setFont(self.font)
        self.back_button.setFixedSize(200, 50)
        layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        #  帮助文档按钮
        self.help_button = QPushButton("help")
        self.help_button.clicked.connect(self.open_readme)
        self.help_button.setStyleSheet(
            '''
            QPushButton {
            background-color: white;
            border-radius: 20px;
            border: 6px solid #C8DBDC;
            padding: 6px;
            color: #83B1C9;

        }
            QPushButton:hover {
        background-color: gray;
        }

        QPushButton:pressed {
        background-color: #E098AE;
        }
            '''
        )
        self.help_button.setFont(self.font)
        self.help_button.setFixedSize(200, 65)
        layout.addWidget(self.help_button, alignment=Qt.AlignCenter)

        # 视频流打开关闭滑动按键的设置
        self.video_stream_slider = QSlider(Qt.Horizontal)  # 设置水平滑扭
        self.video_stream_slider.setRange(0, 1)  # 设置滑扭值的范围，由于是个开关，设置为0，1二值
        self.video_stream_slider.setTickInterval(1)  # 设置刻度间隔
        self.video_stream_slider.setSliderPosition(0)  # 设置滑扭的初始值为0
        self.video_stream_slider.valueChanged.connect(self.video_stream_slider_changed)  # 设置滑扭的值改变时的事件
        self.video_stream_slider.setStyleSheet('''
            QSlider::groove:horizontal {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DDDDDD, stop:1 #DDDDDD);
                border-radius: 14px;
                height: 40px;
            }
            QSlider::sub-page:horizontal {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #65C466, stop:1 #65C466);
                border-radius: 14px;
                height: 100px;
            }
            QSlider::handle:horizontal {
                background-color: #FFFFFF;
                radius: 10px;
                width: 40px;
                height:40px;
                margin: -3px -8px;
                border-radius: 20px;
            }
        ''')  # 设置滑扭的样式
        self.video_stream_slider.setFixedWidth(75)
        self.video_stream_slider.setFixedHeight(50)
        slider_layout = QVBoxLayout()
        slider_layout.addWidget(self.video_stream_label)  # 添加Label到布局
        slider_layout.addWidget(self.video_stream_slider)  # 添加video_stream滑扭到布局

        # 人脸检测打开关闭滑动按键的设置
        self.face_recognition_slider = QSlider(Qt.Horizontal)
        self.face_recognition_slider.setRange(0, 1)
        self.face_recognition_slider.setTickInterval(1)
        self.face_recognition_slider.setSliderPosition(0)
        self.face_recognition_slider.setStyleSheet('''
                    QSlider::groove:horizontal {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DDDDDD, stop:1 #DDDDDD);
                        border-radius: 14px;
                        height: 40px;
                    }
                    QSlider::sub-page:horizontal {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #E098AE, stop:1 #E098AE);
                        border-radius: 14px;
                        height: 40px;
                    }
                    QSlider::handle:horizontal {
                        background-color: #FFFFFF;
                        radius: 20px;
                        width: 40px;
                        height:40px;
                        margin: -3px -8px;
                        border-radius: 20px;
                    }
                ''')
        self.face_recognition_slider.setFixedWidth(75)
        self.face_recognition_slider.setFixedHeight(50)
        self.face_recognition_slider.valueChanged.connect(self.face_recognition_slider_changed)
        slider_layout.addWidget(self.face_recognition_slider)

        # yolo检测打开关闭滑动按键的设置
        self.yolo_recognition_slider = QSlider(Qt.Horizontal)  # 设置水平滑扭
        self.yolo_recognition_slider.setRange(0, 1)  # 设置滑扭值的范围，由于是个开关，设置为0，1二值
        self.yolo_recognition_slider.setTickInterval(1)  # 设置刻度间隔
        self.yolo_recognition_slider.setSliderPosition(0)  # 设置滑扭的初始值为0
        self.yolo_recognition_slider.setStyleSheet('''
                    QSlider::groove:horizontal {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #DDDDDD, stop:1 #DDDDDD);
                        border-radius: 14px;
                        height: 40px;
                    }
                    QSlider::sub-page:horizontal {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FADA5E, stop:1 #FADA5E);
                        border-radius: 14px;
                        height: 100px;
                    }
                    QSlider::handle:horizontal {
                        background-color: #FFFFFF;
                        radius: 10px;
                        width: 40px;
                        height:40px;
                        margin: -3px -8px;
                        border-radius: 20px;
                    }
                ''')  # 设置滑扭的样式
        self.yolo_recognition_slider.setFixedWidth(75)
        self.yolo_recognition_slider.setFixedHeight(50)
        self.yolo_recognition_slider.valueChanged.connect(self.yolo_recognition_slider_changed)
        slider_layout.addWidget(self.yolo_recognition_slider)  # 添加video_stream滑扭到布局

        # yolo识别阈值设置滑块
        self.yolo_threshold_slider = QSlider(Qt.Horizontal)
        self.yolo_threshold_slider.setMaximum(100)
        self.yolo_threshold_slider.setValue(50)
        self.yolo_threshold_slider.setMinimum(0)
        self.yolo_threshold_slider.setSingleStep(1)
        self.yolo_threshold_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 10px;
                width: 500px;
                background-color: #FCE9DA;
                margin: 0px;
            }

            QSlider::handle:horizontal {
                width: 20px;
                height: 20px;
                background-color: #E098AE;
                border-radius: 10px;
                margin: -5px 0;
            }
            """
        )
        self.yolo_threshold_slider.valueChanged.connect(self.yolo_threshold_slider_value_changed)

        # nms识别阈值设置滑块
        self.nms_threshold_slider = QSlider(Qt.Horizontal)
        self.nms_threshold_slider.setMaximum(100)
        self.nms_threshold_slider.setValue(50)
        self.nms_threshold_slider.setMinimum(0)
        self.nms_threshold_slider.setSingleStep(1)
        self.nms_threshold_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 10px;
                width: 500px;
                background-color: #FCE9DA;
                margin: 0px;
            }

            QSlider::handle:horizontal {
                width: 20px;
                height: 20px;
                background-color: #FFCEC7;
                border-radius: 10px;
                margin: -5px 0;
            }
            """
        )
        self.nms_threshold_slider.valueChanged.connect(self.nms_threshold_slider_value_changed)

        # 阈值标签
        self.threshold_value_label = QLabel(f"yolo threshold:{self.yolo_threshold}   "
                                            f"nms threshold:{self.nms_threshold}   ")
        self.threshold_value_label.setStyleSheet("QLabel "
                                                 "{ background-color: #C8DBDC;"
                                                 " border-radius: 20px;"
                                                 " border: 6px solid white; "
                                                 "color: #83B1C9; }"
                                                 )
        self.threshold_value_label.setFont(self.font)
        self.threshold_value_label.setFixedHeight(50)
        # self.threshold_value_label.setFixedWidth(1000)
        self.threshold_value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.yolo_threshold_slider)
        layout.addWidget(self.nms_threshold_slider)
        layout.addWidget(self.threshold_value_label)

        main_layout = QVBoxLayout()  # 主布局
        main_layout.addLayout(layout)  # 添加layout布局
        main_layout.addLayout(slider_layout)  # 添加slider_layout到布局

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        self.video_capture = cv2.VideoCapture(0)  # 打开默认摄像头

        # 加载人脸识别分类器
        self.face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml'
        )
        # 加载人眼识别分类器
        self.eye_cascade = cv2.CascadeClassifier(
            'haarcascade_eye.xml'
        )
        # 加载yolov3网络
        self.yolo_net = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg',
                                                   'yolov4-tiny.weights')  # 读取.cfg 和 .weights文件，获取网络参数
        # self.yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()  # 获取类别名称
        ''' 
        yolov3.cfg 是模型的配置文件，它定义了网络的结构、层的参数以及超参数的设置。
        这个文件描述了 YOLOv3 模型的网络架构，包括卷积层、池化层、全连接层等。
        在这个文件中，可以设置模型的深度、输入图像的尺寸、不同层的过滤器数量等。
        yolov3.weights 是 YOLOv3 模型的预训练权重文件。这个文件包含了经过大规模数据集训练得到的模型参数。
        权重文件是由训练过程中学习到的网络权重组成的二进制文件，其中包含了卷积核、偏置项以及其他层的参数。
        这些权重表示了模型对不同类别的目标的视觉特征的理解。
        coco.names 文件是与 COCO（Common Objects in Context）数据集相关的一个文本文件。
        每一行都包含了一个目标类别的名称，例如 "person"、"car"、"dog" 等。这个文件中的目标类别名称对应于在 COCO 数据集中出现的不同物体类别。
        通过使用 coco.names 文件，可以将模型的输出结果映射到实际的类别名称，从而方便理解和解释模型的预测结果。
        '''
        self.layer_names = self.yolo_net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        self.color_map = {}  # 储存不同的颜色，用于对应coco里边的不同类别
        for label in self.classes:
            color = np.random.randint(0, 255, size=3).tolist()
            self.color_map[label] = color
        '''
        np.random.randint(0, 255, size=3)
        是使用 NumPy 库中的 random.randint() 函数生成一个长度为3的随机整数数组，取值范围为0到255。
        这样生成的数组表示 RGB（红绿蓝）颜色空间中的一个随机颜色，每个元素表示颜色通道的值（红色、绿色和蓝色）。
        '''

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)  # 视频流
        self.video_stream_enabled = False
        self.face_recognition_enabled = False
        self.yolo_recognition_enabled = False

    # 返回电脑摄像头函数
    def back_to_camera(self):
        self.upload = False

    # 上传文件处理函数
    def upload_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if self.file_path:
            if self.file_path.endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.process_image(self.file_path)
            elif self.file_path.endswith((".mp4", ".avi", ".mov")):
                self.upload = True

    # 单张图片处理函数
    def process_image(self, image_path):
        self.upload = True
        self.video_stream_slider_changed(False)
        self.video_stream_slider.setSliderPosition(0)
        frame = cv2.imread(image_path)
        if frame is not None:
            if self.face_recognition_enabled:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 192, 203), 4)  # OpenCV 中，颜色通道的顺序是 BGR
                eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, maxSize=(60, 60))
                for (x, y, w, h) in eyes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            elif self.yolo_recognition_enabled:
                # 构建输入图像的blob
                blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
                # 将blob输入网络进行前向传播
                self.yolo_net.setInput(blob)
                # 前向传播获取检测结果
                outputs = self.yolo_net.forward(self.output_layers)
                # 处理检测结果
                class_ids = []
                confidences = []
                boxes = []
                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > self.yolo_threshold:
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            width = int(detection[2] * frame.shape[1])
                            height = int(detection[3] * frame.shape[0])

                            x = int(center_x - width / 2)
                            y = int(center_y - height / 2)

                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, width, height])

                indices = cv2.dnn.NMSBoxes(list(boxes), confidences, self.yolo_threshold, self.nms_threshold)
                for i in indices:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    label = self.classes[class_id]
                    confidence = confidences[i]
                    color = self.color_map[label]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 10)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                color, 10)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_height, image_width, channel = frame_rgb.shape
            bytes_per_line = channel * image_width
            q_image = QImage(frame_rgb.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(1000, 800, Qt.KeepAspectRatio))

    # 视频流滑动按键事件函数
    def video_stream_slider_changed(self, value):
        if self.upload:
            self.video_capture = cv2.VideoCapture(self.file_path)
            self.fps_time = int(1000/self.video_capture.get(cv2.CAP_PROP_FPS))
        else:
            self.video_capture = cv2.VideoCapture(0)
            self.fps_time = 3
        self.video_stream_enabled = bool(value)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # 设置宽度为1280像素
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280)  # 设置高度为720像素
        if value:
            self.video_text = 'open'
            self.video_stream_slider.setSliderPosition(1)
        else:
            self.video_text = 'close'
            self.video_stream_slider.setSliderPosition(0)
        if self.video_stream_enabled:
            self.timer.start(self.fps_time)  # 设置定时器间隔时间（单位：毫秒）
        else:
            self.timer.stop()
            self.video_capture.release()  # 释放摄像头资源
        self.video_stream_label.setText(f"Video Stream:{self.video_text}  "
                                        f"Face Detection:{self.recognition_text}  "
                                        f"YOLO:{self.yolo_recognition_text}")

    # 人脸识别滑动按键事件函数
    def face_recognition_slider_changed(self, value):
        self.face_recognition_enabled = bool(value)
        if value:
            self.recognition_text = 'open'
            self.yolo_recognition_text = 'close'
            self.yolo_recognition_slider.setSliderPosition(0)
            self.yolo_recognition_enabled = False
            # self.video_stream_slider_changed(self.video_stream_enabled)
        else:
            self.recognition_text = 'close'
            # self.video_stream_slider_changed(self.video_stream_enabled)
        self.video_stream_label.setText(f"Video Stream:{self.video_text}  "
                                        f"Face Detection:{self.recognition_text}  "
                                        f"YOLO:{self.yolo_recognition_text}")

    # yolo识别滑动按键事件函数
    def yolo_recognition_slider_changed(self, value):
        self.yolo_recognition_enabled = bool(value)
        if value:
            self.recognition_text = 'close'
            self.yolo_recognition_text = 'open'
            self.face_recognition_slider.setSliderPosition(0)
            self.face_recognition_enabled = False
            # self.video_stream_slider_changed(self.video_stream_enabled)
        else:
            self.yolo_recognition_text = 'close'
            # self.video_stream_slider_changed(self.video_stream_enabled)
        self.video_stream_label.setText(f"Video Stream:{self.video_text}  "
                                        f"Face Detection:{self.recognition_text}  "
                                        f"YOLO:{self.yolo_recognition_text}")

    # yolo阈值滑动按钮事件函数
    def yolo_threshold_slider_value_changed(self, value):
        self.yolo_threshold = value / 100
        self.threshold_value_label.setText(f'yolo threshold:{self.yolo_threshold}   '
                                           f'nms threshold:{self.nms_threshold}   ')

    # nms阈值滑动按钮事件函数
    def nms_threshold_slider_value_changed(self, value):
        self.nms_threshold = value / 100
        self.threshold_value_label.setText(f"yolo threshold:{self.yolo_threshold}   "
                                           f"nms threshold:{self.nms_threshold}   ")

    # 画图函数
    def display_frame(self):
        ret, frame = self.video_capture.read()
        if ret:
            if self.video_stream_enabled:
                if self.face_recognition_enabled:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 105, 180), 8)  # OpenCV 中，颜色通道的顺序是 BGR
                    eyes = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=10,
                                                             minSize=(10, 10), maxSize=(60, 60))
                    for (x, y, w, h) in eyes:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 192, 203), 2)
                elif self.yolo_recognition_enabled:
                    # 构建输入图像的blob
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), swapRB=True, crop=False)
                    # 将blob输入网络进行前向传播
                    self.yolo_net.setInput(blob)
                    # 前向传播获取检测结果
                    outputs = self.yolo_net.forward(self.output_layers)
                    # 处理检测结果
                    class_ids = []
                    confidences = []
                    boxes = []
                    for output in outputs:
                        for detection in output:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]

                            if confidence > self.yolo_threshold:
                                center_x = int(detection[0] * frame.shape[1])
                                center_y = int(detection[1] * frame.shape[0])
                                width = int(detection[2] * frame.shape[1])
                                height = int(detection[3] * frame.shape[0])

                                x = int(center_x - width / 2)
                                y = int(center_y - height / 2)

                                class_ids.append(class_id)
                                confidences.append(float(confidence))
                                boxes.append([x, y, width, height])

                    indices = cv2.dnn.NMSBoxes(list(boxes), confidences, self.yolo_threshold, self.nms_threshold)
                    for i in indices:
                        x, y, w, h = boxes[i]
                        class_id = class_ids[i]
                        label = self.classes[class_id]
                        confidence = confidences[i]
                        color = self.color_map[label]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 4)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    color, 4)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_height, image_width, channel = frame_rgb.shape
                bytes_per_line = channel * image_width
                q_image = QImage(frame_rgb.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.image_label.setPixmap(pixmap.scaled(1600, 1200, Qt.KeepAspectRatio))

    def open_readme(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile("README.txt"))

    def closeEvent(self, event):
        self.video_capture.release()  # 释放摄像头资源
        event.accept()  # 接受关闭事件


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow { background-color: pink; }")
    app.setWindowIcon(QIcon('L.ico'))
    camera_app = Video_detect()
    camera_app.show()
    sys.exit(app.exec_())

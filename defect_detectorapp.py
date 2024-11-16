import sys
import os
import csv
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QDateTimeEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize
import cv2
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Initialize Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="jxTPOtPBRoa8934S2Chu"
)

# Define font for labels (optional, requires a .ttf file)
try:
    font = ImageFont.truetype("arial.ttf", 18)
except IOError:
    font = ImageFont.load_default()

# Colors
PRIMARY_COLOR = "#5555FF"
SECONDARY_COLOR = "#44BB44"
WARNING_COLOR = "#FF4444"
BACKGROUND_COLOR = "#2b2b2b"
DARK_BG = "#1e1e1e"
TEXT_COLOR = "#ffffff"
HOVER_COLOR = "#4c4cff"
BORDER_COLOR = "#444444"

class AnalysisWindow(QWidget):
    def __init__(self, csv_file):
        super().__init__()
        self.setWindowTitle("Defect Analysis")
        self.showMaximized()  # Make window maximized by default
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")

        # Load data from CSV
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])

        # Filter layout
        filter_layout = QHBoxLayout()
        
        # Initialize with today's date
        self.filter_date = QDateTimeEdit(self)
        self.filter_date.setDateTime(QtCore.QDateTime.currentDateTime())
        self.filter_date.setDisplayFormat("yyyy-MM-dd")
        self.filter_date.setCalendarPopup(True)
        self.filter_date.setStyleSheet(f"""
            background-color: {DARK_BG}; color: {TEXT_COLOR}; border-radius: 5px; padding: 5px;
            font-size: 14px; border: 1px solid {BORDER_COLOR};
        """)
        
        filter_button = QPushButton("Filter")
        filter_button.setStyleSheet(f"""
            background-color: {PRIMARY_COLOR}; color: {TEXT_COLOR};
            padding: 12px; font-size: 16px; border-radius: 5px;
            border: 1px solid {BORDER_COLOR};
        """)
        filter_button.setCursor(Qt.PointingHandCursor)
        filter_button.clicked.connect(self.update_analysis)

        filter_layout.addWidget(QLabel("Filter Date:", alignment=Qt.AlignRight), stretch=1)
        filter_layout.addWidget(self.filter_date, stretch=2)
        filter_layout.addWidget(filter_button, stretch=1)
        filter_layout.addStretch()

        # Create layout for charts and list
        content_layout = QHBoxLayout()

        # Chart area
        chart_layout = QVBoxLayout()
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background-color: {DARK_BG}; border-radius: 8px;")
        chart_layout.addWidget(self.canvas)

        # Defect summary list
        list_layout = QVBoxLayout()
        list_label = QLabel("Defect Summary")
        list_label.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        self.defect_summary_list = QListWidget()
        self.defect_summary_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                font-size: 14px;
                padding: 5px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            QListWidget::item {{
                padding: 10px;
                border-bottom: 1px solid {BORDER_COLOR};
            }}
            QListWidget::item:hover {{
                background-color: {HOVER_COLOR};
            }}
        """)
        list_layout.addWidget(list_label)
        list_layout.addWidget(self.defect_summary_list)

        # Add chart and list to content layout
        content_layout.addLayout(chart_layout, stretch=2)
        content_layout.addLayout(list_layout, stretch=1)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(filter_layout)
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # Show initial analysis
        self.update_analysis()

    def update_analysis(self):
        # Get selected date
        filter_date = self.filter_date.date().toPyDate()
        
        # Filter data for the selected date
        filtered_df = self.df[self.df['Timestamp'].dt.date == filter_date]

        # Count occurrences of each defect
        defect_counts = filtered_df['Defect Type'].value_counts()

        # Update plot
        self.ax.clear()
        if not defect_counts.empty:
            defect_counts.plot(kind='bar', color=PRIMARY_COLOR, ax=self.ax)
            self.ax.set_title(f"Defect Frequency Analysis - {filter_date.strftime('%Y-%m-%d')}", fontsize=16, color=TEXT_COLOR)
            self.ax.set_xlabel("Defect Type", fontsize=14, color=TEXT_COLOR)
            self.ax.set_ylabel("Frequency", fontsize=14, color=TEXT_COLOR)
            self.ax.tick_params(axis='x', rotation=45, labelcolor=TEXT_COLOR)  # Set tick label color to white
            self.ax.tick_params(axis='y', labelcolor=TEXT_COLOR)  # Set y-axis label color to white
            # Set the color of the bars' labels
            for label in self.ax.get_xticklabels():
                label.set_color(TEXT_COLOR)  # Set x-axis labels to white
            for label in self.ax.get_yticklabels():
                label.set_color(TEXT_COLOR)  # Set y-axis labels to white
        else:
            self.ax.text(0.5, 0.5, "No data available for selected date", ha='center', va='center', fontsize=14, color=TEXT_COLOR)
        
        # Set background and figure color
        self.ax.set_facecolor(DARK_BG)
        self.figure.patch.set_facecolor(BACKGROUND_COLOR)
        
        # Redraw the canvas
        self.canvas.draw()

        # Update summary list
        self.defect_summary_list.clear()
        if not defect_counts.empty:
            self.defect_summary_list.addItem("Total Defects by Type:")
            for defect_type, count in defect_counts.items():
                self.defect_summary_list.addItem(f"• {defect_type}: {count}")
            
            # Add total count
            total_defects = defect_counts.sum()
            self.defect_summary_list.addItem("")
            self.defect_summary_list.addItem(f"Total Defects: {total_defects}")
        else:
            self.defect_summary_list.addItem("No defects found for selected date")


class VideoInferenceApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main Window Configuration
        self.setWindowTitle("PCB Defect Detection")
        self.setGeometry(100, 100, 1400, 800)
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR};")

        # Layouts and Widgets
        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Video Display Label (Larger Fixed Size Box)
        self.video_label = QLabel()
        self.video_label.setFixedSize(1000, 600)
        self.video_label.setStyleSheet(f"background-color: {DARK_BG}; border: 2px solid {BORDER_COLOR}; border-radius: 8px;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # Buttons
        button_style = f"""
            background-color: {PRIMARY_COLOR}; color: {TEXT_COLOR};
            padding: 12px; font-size: 16px; border-radius: 5px;
            border: 1px solid {BORDER_COLOR};
        """

        self.open_button = QPushButton("Open Video")
        self.open_button.setStyleSheet(button_style)
        self.open_button.setCursor(Qt.PointingHandCursor)
        self.open_button.clicked.connect(self.open_video)

        self.snapshot_button = QPushButton("Take Snapshot")
        self.snapshot_button.setStyleSheet(f"background-color: {SECONDARY_COLOR}; color: {TEXT_COLOR}; padding: 12px; font-size: 16px; border-radius: 5px; border: 1px solid {BORDER_COLOR};")
        self.snapshot_button.setCursor(Qt.PointingHandCursor)
        self.snapshot_button.clicked.connect(self.save_snapshot)

        self.open_folder_button = QPushButton("Open Images Folder")
        self.open_folder_button.setStyleSheet(f"background-color: {WARNING_COLOR}; color: {TEXT_COLOR}; padding: 12px; font-size: 16px; border-radius: 5px; border: 1px solid {BORDER_COLOR};")
        self.open_folder_button.setCursor(Qt.PointingHandCursor)
        self.open_folder_button.clicked.connect(self.open_images_folder)

        # New Button for Analyzing CSV
        self.analyze_button = QPushButton("Analyze CSV")
        self.analyze_button.setStyleSheet(button_style)
        self.analyze_button.setCursor(Qt.PointingHandCursor)
        self.analyze_button.clicked.connect(self.open_analysis_window)

        # Sidebar for Defect List
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(self.open_button)
        sidebar_layout.addWidget(self.snapshot_button)
        sidebar_layout.addWidget(self.open_folder_button)
        sidebar_layout.addWidget(self.analyze_button)
        self.defect_list = QListWidget()
        self.defect_list.setFixedWidth(300)
        self.defect_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {DARK_BG};
                color: {TEXT_COLOR};
                padding: 10px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
            }}
            QListWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {BORDER_COLOR};
            }}
            QListWidget::item:hover {{
                background-color: {HOVER_COLOR};
            }}
        """)
        sidebar_layout.addWidget(QLabel("Current Detected Defects", alignment=Qt.AlignCenter, styleSheet="font-size: 16px; color: #ffffff;"))
        sidebar_layout.addWidget(self.defect_list)

        # Horizontal Layout to hold video and sidebar
        video_sidebar_layout = QHBoxLayout()
        video_sidebar_layout.addWidget(self.video_label)
        video_sidebar_layout.addLayout(sidebar_layout)

        main_layout.addLayout(video_sidebar_layout)

        # Video Capture and Processing Variables
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.frame_skip = 60
        self.frame_count = 0

        # Folder for saving snapshots
        self.images_folder = "images"
        os.makedirs(self.images_folder, exist_ok=True)

        self.csv_file = "defect_data.csv"
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Defect Type"])

    def open_video(self):
        # Open file dialog to select video file
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            self.cap = cv2.VideoCapture(file_path)
            self.frame_count = 0
            self.defect_list.clear()
            self.timer.start(30)  # Start timer for updating frames

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return

            # Skip frames for smoother display
            if self.frame_count % self.frame_skip == 0:
                resized_frame = cv2.resize(frame, (640, 360))
                frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Perform inference on the resized frame
                result = CLIENT.infer(pil_image, model_id="pcb-defect-detection-ryztp/1")
                draw = ImageDraw.Draw(pil_image)

                # Clear defect list for current frame
                self.defect_list.clear()

                # Annotate each prediction and save to CSV
                unique_defects = set()
                for prediction in result['predictions']:
                    x = prediction['x'] * (640 / frame.shape[1])
                    y = prediction['y'] * (360 / frame.shape[0])
                    width = prediction['width'] * (640 / frame.shape[1])
                    height = prediction['height'] * (360 / frame.shape[0])
                    confidence = prediction['confidence']
                    class_name = prediction['class']

                    # Draw bounding box and label
                    left = x - width / 2
                    top = y - height / 2
                    right = x + width / 2
                    bottom = y + height / 2
                    draw.rectangle([left, top, right, bottom], outline="red", width=2)

                    # Draw the label
                    label = f"{class_name}: {confidence:.2f}"
                    draw.text((left, top - 20), label, fill="white", font=font)

                    # Add defect to sidebar list
                    self.defect_list.addItem(f"{class_name} (Confidence: {confidence:.2f})")

                    # Collect unique defects
                    unique_defects.add(class_name)

                # Save detected defects to CSV file
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    for defect in unique_defects:
                        writer.writerow([timestamp, defect])

                # Convert annotated frame back to OpenCV format
                annotated_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))

                # Convert OpenCV frame to Qt format and display
                qt_image = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], QImage.Format_BGR888)
                pixmap = QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(pixmap)

            self.frame_count += 1

    def save_snapshot(self):
        if self.cap and self.cap.isOpened():
            snapshot_path = os.path.join(self.images_folder, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            pixmap = self.video_label.pixmap()
            if pixmap:
                pixmap.save(snapshot_path)
                print(f"Snapshot saved at {snapshot_path}")

    def open_images_folder(self):
        os.startfile(self.images_folder)

    def open_analysis_window(self):
        self.analysis_window = AnalysisWindow(self.csv_file)
        self.analysis_window.show()

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

# Run the application
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = VideoInferenceApp()
    window.show()
    sys.exit(app.exec_())

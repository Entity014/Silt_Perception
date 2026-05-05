import sys
import cv2
import os
import csv
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
                             QListWidget, QSplitter)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

class ClickableImageLabel(QLabel):
    # ส่งพิกัด x, y เมื่อมีการคลิก
    clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.original_cv_image = None
        self.display_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #2b2b2b;")

    def set_cv_image(self, cv_img):
        self.original_cv_image = cv_img
        self.update_pixmap()

    def update_pixmap(self):
        if self.original_cv_image is None:
            self.clear()
            return
        
        # แปลง BGR (OpenCV) เป็น RGB (PyQt)
        rgb_image = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(q_img)
        
        # ปรับขนาดรูปให้พอดีกับหน้าต่าง (รักษาสัดส่วน)
        scaled_pix = pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.display_pixmap = scaled_pix
        self.setPixmap(self.display_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.original_cv_image is not None:
            self.update_pixmap()

    def mousePressEvent(self, event):
        if self.original_cv_image is None or self.display_pixmap is None:
            return
            
        label_w = self.width()
        label_h = self.height()
        pix_w = self.display_pixmap.width()
        pix_h = self.display_pixmap.height()
        
        # คำนวณ Offset เพราะเราจัดให้อยู่กึ่งกลาง
        offset_x = (label_w - pix_w) / 2.0
        offset_y = (label_h - pix_h) / 2.0
        
        click_x = event.position().x() - offset_x
        click_y = event.position().y() - offset_y
        
        # ตรวจสอบว่าคลิกโดนรูปภาพจริงๆ ไม่ใช่ขอบดำ
        if 0 <= click_x <= pix_w and 0 <= click_y <= pix_h:
            # คำนวณกลับไปเป็นพิกัดจริงของรูปภาพ (Original Resolution)
            orig_h, orig_w = self.original_cv_image.shape[:2]
            real_x = int(click_x * orig_w / pix_w)
            real_y = int(click_y * orig_h / pix_h)
            self.clicked.emit(real_x, real_y)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Silt Prediction GUI (OpenCV + PyQt)")
        self.resize(1000, 700)
        
        self.data_dir = None
        self.image_files = []
        self.current_index = -1
        self.csv_data = {} # dictionary เพื่อเก็บข้อมูล {filename: {upper_x, upper_y, lower_x, lower_y}}
        
        self.current_points = []
        self.current_raw_image = None
        
        self.initUI()
        
    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # ส่วนซ้าย: ควบคุม และ รายการไฟล์
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0,0,0,0)
        
        btn_open = QPushButton("Open File (Select Image)")
        btn_open.clicked.connect(self.open_file)
        btn_open.setStyleSheet("padding: 8px; font-weight: bold;")
        
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self.load_image_by_index)
        
        btn_prev = QPushButton("< Previous")
        btn_prev.clicked.connect(self.prev_image)
        
        btn_next = QPushButton("Next >")
        btn_next.clicked.connect(self.next_image)
        
        btn_clear = QPushButton("Clear Points (Current Image)")
        btn_clear.clicked.connect(self.clear_points)
        
        btn_save = QPushButton("Save All to CSV")
        btn_save.clicked.connect(self.save_csv)
        btn_save.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(btn_prev)
        nav_layout.addWidget(btn_next)
        
        self.info_label = QLabel("Water Top: None\nWater Bottom: None\nSilt Top: None")
        self.info_label.setStyleSheet("font-size: 14px; padding: 10px; color: #ffffff;")
        
        left_layout.addWidget(btn_open)
        left_layout.addWidget(QLabel("Image List:"))
        left_layout.addWidget(self.list_widget)
        left_layout.addLayout(nav_layout)
        left_layout.addWidget(btn_clear)
        left_layout.addWidget(self.info_label)
        left_layout.addStretch()
        left_layout.addWidget(btn_save)
        
        # ส่วนขวา: แสดงรูปภาพ
        self.image_label = ClickableImageLabel()
        self.image_label.clicked.connect(self.on_image_clicked)
        
        # ใช้ Splitter เผื่อผู้ใช้ต้องการปรับขนาด Sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.image_label)
        splitter.setSizes([250, 750]) # กำหนดขนาดเริ่มต้น (ซ้ายเล็ก ขวาใหญ่)
        
        main_layout.addWidget(splitter)
        
    def open_file(self):
        # เปิดหน้าต่างให้เลือกรูปภาพ
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not file_path:
            return
            
        self.data_dir = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # ค้นหารูปทั้งหมดที่อยู่ในโฟลเดอร์เดียวกัน
        valid_ext = ('.png', '.jpg', '.jpeg')
        self.image_files = sorted([f for f in os.listdir(self.data_dir) if f.lower().endswith(valid_ext)])
        
        self.list_widget.clear()
        self.list_widget.addItems(self.image_files)
        
        # โหลดข้อมูล CSV เก่า (ถ้ามี)
        self.load_csv()
        
        # เลือกไฟล์ที่ถูกคลิก
        try:
            index = self.image_files.index(filename)
            self.list_widget.setCurrentRow(index)
        except ValueError:
            pass

    def load_csv(self):
        self.csv_data.clear()
        csv_file = os.path.join(self.data_dir, "ground_truth.csv")
        if os.path.exists(csv_file):
            try:
                with open(csv_file, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        fname = row.get('filename')
                        if fname:
                            self.csv_data[fname] = {
                                'water_top_y': int(row.get('water_top_y', 0)),
                                'water_bottom_y': int(row.get('water_bottom_y', 0)),
                                'silt_top_y': int(row.get('silt_top_y', 0)),
                            }
            except Exception as e:
                QMessageBox.warning(self, "CSV Error", f"Failed to load CSV: {e}")

    def save_csv(self):
        if not self.data_dir:
            return
            
        # บันทึกสถานะรูปล่าสุดเข้า Dictionary ก่อน
        self.commit_current_points()
            
        csv_file = os.path.join(self.data_dir, "ground_truth.csv")
        try:
            with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'water_top_y', 'water_bottom_y', 'silt_top_y'])
                
                # เขียนข้อมูลเฉพาะไฟล์ที่มีการกำหนดจุด 3 จุดแล้ว
                for fname in self.image_files:
                    if fname in self.csv_data:
                        d = self.csv_data[fname]
                        writer.writerow([fname, d['water_top_y'], d['water_bottom_y'], d['silt_top_y']])
                        
            QMessageBox.information(self, "Success", f"บันทึกข้อมูลลงไฟล์\n{csv_file}\nเรียบร้อยแล้ว!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"ไม่สามารถบันทึก CSV ได้: {e}")

    def load_image_by_index(self, index):
        if index < 0 or index >= len(self.image_files):
            return
            
        # ก่อนเปลี่ยนรูป ให้จำจุดของรูปเดิมไว้ก่อน (ถ้าจุดครบ)
        if self.current_index != -1 and self.current_index != index:
            self.commit_current_points()
            
        self.current_index = index
        filename = self.image_files[index]
        image_path = os.path.join(self.data_dir, filename)
        
        # ใช้ OpenCV อ่านภาพ เพื่อเก็บไว้ทำ Image Processing ในอนาคต
        self.current_raw_image = cv2.imread(image_path)
        if self.current_raw_image is None:
            self.image_label.clear()
            self.setWindowTitle(f"Error loading {filename}")
            return
            
        self.setWindowTitle(f"Silt Prediction - {filename}")
        
        # ดึงจุดเก่าจาก CSV (ถ้ามี)
        self.current_points = []
        if filename in self.csv_data:
            d = self.csv_data[filename]
            # ใช้ x สมมติ (กลางภาพ) เพราะเราไม่ได้เซฟ x
            self.current_points.append((100, d['water_top_y']))
            self.current_points.append((100, d['water_bottom_y']))
            self.current_points.append((100, d['silt_top_y']))
            
        self.update_display()
        
    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.list_widget.setCurrentRow(self.current_index + 1)
            
    def prev_image(self):
        if self.current_index > 0:
            self.list_widget.setCurrentRow(self.current_index - 1)
            
    def clear_points(self):
        # ล้างจุดในรูปปัจจุบัน
        self.current_points = []
        filename = self.image_files[self.current_index]
        if filename in self.csv_data:
            del self.csv_data[filename]
        self.update_display()
        
    def commit_current_points(self):
        # บันทึกจุดของรูปปัจจุบันลง Dictionary (เฉพาะกรณีที่เลือกครบ 2 จุดแล้ว)
        if self.current_index >= 0 and self.current_index < len(self.image_files):
            filename = self.image_files[self.current_index]
            if len(self.current_points) == 3:
                self.csv_data[filename] = {
                    'water_top_y': self.current_points[0][1],
                    'water_bottom_y': self.current_points[1][1],
                    'silt_top_y': self.current_points[2][1],
                }

    def on_image_clicked(self, x, y):
        # ถ้าจิ้มครบ 3 จุดแล้ว ให้รีเซ็ตใหม่เพื่อเริ่มจิ้มใหม่
        if len(self.current_points) >= 3:
            self.current_points = [] 
            
        self.current_points.append((x, y))
        self.update_display()

    def update_display(self):
        if self.current_raw_image is None:
            return
            
        # สร้าง Copy ของภาพเพื่อวาดเส้นทับ (จะไม่แก้ไขภาพต้นฉบับ)
        display_img = self.current_raw_image.copy()
        h, w = display_img.shape[:2]
        
        w_top_str = "None"
        w_bottom_str = "None"
        silt_top_str = "None"
        
        # 1. Water Top (Green)
        if len(self.current_points) >= 1:
            ux, uy = self.current_points[0]
            cv2.line(display_img, (0, uy), (w, uy), (0, 255, 0), 2)
            cv2.circle(display_img, (ux, uy), 5, (0, 255, 0), -1)
            cv2.putText(display_img, f"Water Top Y={uy}", (10, uy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            w_top_str = f"{uy}"
            
        # 2. Water Bottom (Blue)
        if len(self.current_points) >= 2:
            lx, ly = self.current_points[1]
            cv2.line(display_img, (0, ly), (w, ly), (255, 0, 0), 2) # Blue in BGR
            cv2.circle(display_img, (lx, ly), 5, (255, 0, 0), -1)
            cv2.putText(display_img, f"Water Bottom Y={ly}", (10, ly - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            w_bottom_str = f"{ly}"

        # 3. Silt Top (Red)
        if len(self.current_points) >= 3:
            sx, sy = self.current_points[2]
            cv2.line(display_img, (0, sy), (w, sy), (0, 0, 255), 2) # Red in BGR
            cv2.circle(display_img, (sx, sy), 5, (0, 0, 255), -1)
            cv2.putText(display_img, f"Silt Top Y={sy}", (10, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            silt_top_str = f"{sy}"
            
        self.info_label.setText(f"Water Top: {w_top_str}\nWater Bottom: {w_bottom_str}\nSilt Top: {silt_top_str}")
        
        # ส่งภาพที่วาดเส้นแล้วไปแสดงผลใน PyQt Label
        self.image_label.set_cv_image(display_img)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

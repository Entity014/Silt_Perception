# Silt Edge Detection Project

ระบบตรวจวัดระดับตะกอนอัตโนมัติจากรูปภาพ โดยใช้เทคนิค Computer Vision และการประมวลผลภาพแบบลำดับขั้นตอน (Pipeline) เพื่อหาปริมาตรตะกอนในกรวย Imhoff (Imhoff Cone)

## คุณสมบัติเด่น (Features)

- **OOP Structure**: พัฒนาด้วยโครงสร้างแบบ Object-Oriented เพื่อความง่ายในการขยายและนำไปใช้งานต่อ
- **Automated Pipeline**:
  - **Auto Light Balance (CLAHE)**: ปรับสมดุลแสงในภาพโดยอัตโนมัติ
  - **Tube Projection**: ค้นหาขอบเขตของหลอดทดลอง (ROI) อัตโนมัติ
  - **Segmentation**: ใช้เทคนิค GrabCut และ K-Means Clustering (Auto-K) เพื่อแยกตะกอนออกจากพื้นหลัง
  - **Probabilistic Detection**: ค้นหาขอบบนของตะกอนโดยใช้หลักความน่าจะเป็นจากตำแหน่งและความหนาแน่น
- **Volume Estimation**: คำนวณปริมาตรในหน่วยมิลลิลิตร (ml) ตามรูปทรงของกรวย
- **Evaluation System**: เปรียบเทียบผลลัพธ์กับ Ground Truth และคำนวณค่า Mean Absolute Error (MAE)
- **Result Export**: บันทึกผลการตรวจวัดลงในไฟล์ CSV อัตโนมัติ (`detection_results.csv`) พร้อมภาพแยกแต่ละขั้นตอนในโฟลเดอร์ `assets/`

## Pipeline & หลักการทำงาน (Algorithm Flow)

1. **Auto Light Balance (CLAHE)**: ปรับความเปรียบต่างและสมดุลแสงของภาพให้สม่ำเสมอ
2. **Tube Projection (ROI Detection)**: หาขอบซ้าย-ขวาของหลอดแก้วอัตโนมัติโดยใช้ Sobel X และ Moving Average
3. **GrabCut Segmentation**: กำจัดพื้นหลังที่อยู่นอกหลอดแก้วออก
4. **Auto K-Means Clustering**: จัดกลุ่มสีและหาจำนวนกลุ่ม (K) ที่เหมาะสมที่สุดด้วย Elbow Method เพื่อแยกชั้นตะกอน
5. **Silt Mask Generation**: สร้าง Binary Mask จากกลุ่มสีที่มืดที่สุด (ตะกอน)
6. **Y-Histogram & Probabilistic Detection**: หาขอบบนสุดของตะกอนโดยใช้กราฟความหนาแน่นและถ่วงน้ำหนักตามตำแหน่ง (เน้นพิกเซลที่อยู่ด้านล่างของภาพ)
7. **Volume Conversion**: แปลงความสูงของตะกอน (Pixels) เป็นปริมาตร (ml) โดยอิงจากระดับน้ำจริง

## การปรับแต่งพารามิเตอร์ (Parameters)

คุณสามารถปรับแต่งการทำงานได้ผ่าน `DEFAULT_CONFIG` ในไฟล์ `main.py`:

### 1. Pipeline Settings

| Parameter                | Default                   | Description                                                           |
| ------------------------ | ------------------------- | --------------------------------------------------------------------- |
| `data_dir`               | `"data"`                  | Directory containing input images.                                    |
| `gt_file`                | `"data/ground_truth.csv"` | Ground truth CSV file path for performance evaluation.                |
| `output_file`            | `"detection_results.csv"` | Path for the summary results CSV file.                                |
| `max_width`              | `600`                     | Maximum image width for processing (improves performance).            |
| `save_linear_comparison` | `True`                    | Toggle for calculating and saving linear vs. cone volume estimations. |

### 2. Algorithm Parameters

| Parameter              | Default   | Priority      | Description                                                            |
| ---------------------- | --------- | ------------- | ---------------------------------------------------------------------- |
| `vol_exponent`         | `2.3`     | Optional      | Exponent for cone-shaped volume calculation.                           |
| `vol_exponent_linear`  | `1.0`     | Optional      | Exponent for linear volume calculation.                                |
| `clahe_clip_limit`     | `2.0`     | Optional      | Contrast limit for CLAHE light balancing.                              |
| `proj_window`          | `10`      | **Essential** | Moving average window size for tube boundary detection.                |
| `proj_threshold_ratio` | `0.2`     | **Essential** | Threshold for tube edge detection (relative to peak intensity).        |
| `k_min`, `k_max`       | `3`, `10` | **Essential** | Range of clusters (K) for K-Means elbow method.                        |
| `k_offset`             | `2`       | **Essential** | Additional offset added to the optimal K found by the elbow method.    |
| `grabcut_iters`        | `5`       | Optional      | Number of iterations for the GrabCut foreground extraction.            |
| `mask_threshold`       | `45`      | Optional      | Initial threshold for creating the silt mask (if K-Means is not used). |
| `y_threshold_ratio`    | `0.1`     | **Essential** | Minimum density threshold to identify silt pixel clusters.             |
| `pos_weight_exponent`  | `2`       | Optional      | Exponent for position-based weighting.                                 |

## โครงสร้างโปรเจกต์

- `main.py`: ส่วนบริหารจัดการการทำงานหลัก (Entry Point)
- `lib/silt_detector.py`: โมดูลประมวลผลภาพ (SiltDetector Class)
- `data/`: เก็บรูปภาพและไฟล์ `ground_truth.csv`
- `assets/`: เก็บภาพผลลัพธ์แยกตามขั้นตอน (สร้างขึ้นอัตโนมัติ)
- `requirements.txt`: ไฟล์ระบุ Library ที่จำเป็น

## การติดตั้งและใช้งาน

1. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. รันโปรแกรม:
   ```bash
   python3 main.py
   ```
   _หมายเหตุ: ปรับ `main(visualize=True, save_assets=True)` ในส่วนท้ายของไฟล์เพื่อเปิด/ปิดการแสดงผลภาพและบันทึก Assets_

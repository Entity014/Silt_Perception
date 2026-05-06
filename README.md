# Silt Edge Detection Project

ระบบตรวจวัดระดับตะกอนอัตโนมัติจากรูปภาพ โดยใช้เทคนิค Computer Vision และการประมวลผลภาพแบบลำดับขั้นตอน (Pipeline)

## คุณสมบัติเด่น (Features)

- **OOP Structure**: พัฒนาด้วยโครงสร้างแบบ Object-Oriented เพื่อความง่ายในการขยายและนำไปใช้งานต่อ
- **Automated Pipeline**:
  - **Auto Light Balance (CLAHE)**: ปรับสมดุลแสงในภาพโดยอัตโนมัติ
  - **Tube Projection**: ค้นหาขอบเขตของหลอดทดลอง (ROI) อัตโนมัติ
  - **Segmentation**: ใช้เทคนิค GrabCut และ K-Means Clustering (Auto-K) เพื่อแยกตะกอนออกจากพื้นหลัง
  - **Probabilistic Detection**: ค้นหาขอบบนของตะกอนโดยใช้หลักความน่าจะเป็นจากตำแหน่งและความหนาแน่น
- **Evaluation System**: เปรียบเทียบผลลัพธ์กับ Ground Truth และคำนวณค่า Mean Absolute Error (MAE)
- **Result Export**: บันทึกผลการตรวจวัดลงในไฟล์ CSV อัตโนมัติ (`detection_results.csv`)

## Pipeline & หลักการทำงาน (Algorithm Flow)

โปรแกรมใช้เทคนิค Computer Vision ประมวลผลภาพตามลำดับขั้นตอนเพื่อความแม่นยำ:

1. **Auto Light Balance (CLAHE)**: 
   ปรับความเปรียบต่างและสมดุลแสงของภาพให้สม่ำเสมอ ลดปัญหาแสงสะท้อนและเงาน้ำ
2. **Tube Projection (ROI Detection)**: 
   หาขอบซ้าย-ขวาของหลอดแก้วอัตโนมัติ โดยใช้ **Sobel X Edge Detection** หาเส้นขอบแนวตั้ง นำมาบวกทบกัน (Column Projection) และเกลี่ยกราฟด้วย **Moving Average** เพื่อหาตำแหน่งขอบหลอดที่ชัดเจนที่สุด
3. **GrabCut Segmentation**: 
   กำจัดพื้นหลัง (Background) ที่อยู่นอกหลอดแก้วออกทั้งหมด ให้เหลือเฉพาะของเหลวและตะกอน
4. **Auto K-Means Clustering**: 
   จัดกลุ่มสี (Color Quantization) แบบอัตโนมัติ โดยใช้ **Elbow Method** หาจำนวนกลุ่ม (K) ที่เหมาะสมที่สุด เพื่อแยกชั้นสีของน้ำและตะกอนให้ตัดกันชัดเจน
5. **Silt Mask Generation**: 
   สร้าง Binary Mask โดยดึงเฉพาะสีที่เป็นตะกอน (หาจากโหมดสีที่มืดที่สุด) พร้อมใช้ **Morphological Operations** เปิดปิดภาพเพื่อลบจุดรบกวนเล็กๆ (Noise)
6. **Y-Histogram & Probabilistic Detection**: 
   รวมพิกเซลของตะกอนในแกน Y เพื่อสร้างกราฟความหนาแน่น (Y-Histogram) จากนั้นคำนวณหาคลัสเตอร์ของตะกอนที่ใหญ่ที่สุดและหา **ขอบบนสุดของตะกอน (Silt Edge)** ด้วยหลักความน่าจะเป็น

## โครงสร้างโปรเจกต์

- `main.py`: ส่วนบริหารจัดการการทำงานหลัก (Application Entry Point) และการแสดงผล
- `lib/silt_detector.py`: โมดูลหลักที่รวบรวมอัลกอริทึมการทำ `Image Processing` และคลาส `SiltDetector`
- `data/`: โฟลเดอร์สำหรับเก็บรูปภาพทดสอบและไฟล์ Ground Truth (`ground_truth.csv`)
- `requirements.txt`: ไฟล์ระบุ `library` ที่จำเป็นสำหรับการรันโปรเจกต์

## การติดตั้ง (Setup)

1. สร้าง Virtual Environment:
   ```bash
   python3 -m venv venv
   ```
2. เปิดใช้งาน Virtual Environment:
   - Linux/macOS: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
3. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## การใช้งาน (Usage)

รันสคริปต์หลัก:

```bash
python3 main.py
```

### การปรับแต่ง

คุณสามารถปรับแต่งการแสดงผลได้ใน `main.py`:

```python
if __name__ == "__main__":
    main(visualize=True) # เปลี่ยนเป็น False เพื่อปิดการแสดงหน้าต่างภาพ (ทำงานแบบเงียบ)
```

## การประเมินผล (Evaluation)

หากมีไฟล์ `data/ground_truth.csv` ระบบจะแสดงเส้นประสีเขียวเป็นค่าจริง (Ground Truth) และคำนวณ Error เมื่อสิ้นสุดการทำงาน

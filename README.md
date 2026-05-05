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

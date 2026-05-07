import cv2
import numpy as np
import os
import csv
from lib.silt_detector import SiltDetector


def main(visualize=True, save_assets=True):
    data_dir = "data"
    gt_file = os.path.join(data_dir, "ground_truth.csv")
    output_file = "detection_results.csv"
    valid_ext = ('.jpg', '.jpeg', '.png')
    
    
    if not os.path.exists(data_dir):
        print(f"ไม่พบโฟลเดอร์ {data_dir}")
        return
        
    image_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_ext)])
    
    if not image_files:
        print("ไม่พบรูปภาพในโฟลเดอร์ data")
        return

    # โหลด Ground Truth
    ground_truth = {}
    if os.path.exists(gt_file):
        with open(gt_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ground_truth[row['filename']] = {
                        'water_top_y': int(row['water_top_y']),
                        'water_bottom_y': int(row['water_bottom_y']),
                        'silt_top_y': int(row['silt_top_y']),
                        'silt_base': float(row['silt_base'])
                    }
                except (ValueError, KeyError):
                    continue
        print(f"โหลดข้อมูล Ground Truth สำเร็จ ({len(ground_truth)} รายการ)")
    else:
        print("คำเตือน: ไม่พบไฟล์ ground_truth.csv จะข้ามขั้นตอนการ Evaluation")

    print("--------------------------------------------------")
    print("Silt Edge Detection Pipeline (OOP Version)")
    print("  CLAHE → ROI → GrabCut → K-Means → Silt Detection")
    if visualize:
        print(" - กดปุ่มใดๆ เพื่อดูรูปถัดไป")
        print(" - กด 'q' เพื่อออก")
    print("--------------------------------------------------")

    detector = SiltDetector()
    if visualize:
        window_name = "Silt Detection Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 1200, 800)

    errors = []
    vol_errors = []
    results_data = []

    for idx, filename in enumerate(image_files):
        filepath = os.path.join(data_dir, filename)
        img = cv2.imread(filepath)
        
        if img is None:
            continue
            
        height, width = img.shape[:2]
        scale = 1.0
        if width > 600:
            scale = 600 / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            height, width = img.shape[:2] # Update height/width after resize
            
        # ประมวลผลภาพโดยใช้ SiltDetector
        results = detector.process(img)
        
        silt_edge_y = results['silt_edge_y']
        balanced_img = results['balanced_img']
        lx = results['lx']
        rx = results['rx']
        y_hist_display = results['y_hist_display']
        y_clusters = results['y_clusters']
        cluster_probs = results['cluster_probs']
        best_cluster_idx = results['best_cluster_idx']

        # คำนวณค่า Pred ใน Original Scale เพื่อเซฟลง CSV
        pred_orig = int(silt_edge_y / scale) if silt_edge_y is not None else ""
        
        pred_vol = ""
        if filename in ground_truth and pred_orig != "":
            gt_row = ground_truth[filename]
            pred_vol = detector.estimate_volume(pred_orig, gt_row['water_top_y'], gt_row['water_bottom_y'])
        
        # เตรียมข้อมูลสำหรับเซฟลง CSV (ให้เหมือน GT แต่เพิ่ม pred_pixel และ pred_volume)
        row = {
            'filename': filename,
            'water_top_y': ground_truth[filename]['water_top_y'] if filename in ground_truth else "",
            'water_bottom_y': ground_truth[filename]['water_bottom_y'] if filename in ground_truth else "",
            'silt_top_y': ground_truth[filename]['silt_top_y'] if filename in ground_truth else "",
            'silt_base': ground_truth[filename]['silt_base'] if filename in ground_truth else "",
            'pred_pixel': pred_orig,
            'pred_volume': pred_vol
        }
        results_data.append(row)

        # เปรียบเทียบกับ Ground Truth (Evaluation Logic)
        eval_text = ""
        if filename in ground_truth:
            gt_silt_y = int(ground_truth[filename]['silt_top_y'] * scale)
            gt_water_top_y = int(ground_truth[filename]['water_top_y'] * scale)
            gt_water_bottom_y = int(ground_truth[filename]['water_bottom_y'] * scale)
            
            # ใช้ silt_base จาก GT เป็นปริมาณอ้างอิง (ถ้ามี)
            gt_vol_vis = ground_truth[filename].get('silt_base')
            if gt_vol_vis is None:
                gt_vol_vis = detector.estimate_volume(gt_silt_y, gt_water_top_y, gt_water_bottom_y)
            
            if silt_edge_y is not None:
                error = abs(silt_edge_y - gt_silt_y)
                errors.append(error)
                if gt_vol_vis is not None and isinstance(pred_vol, (int, float)):
                    error_vol = abs(pred_vol - gt_vol_vis)
                    vol_errors.append(error_vol)
                    eval_text = f" | GT: {gt_vol_vis:.1f} ml | Err: {error_vol:.1f} ml"
                else:
                    eval_text = f" | GT: {gt_silt_y} px | Error: {error} px"

        # Visualization (OOP)
        if visualize or save_assets:
            gt_data = ground_truth.get(filename)
            canvas, panel_images = detector.create_visualization(
                img, results, pred_vol=pred_vol, gt_data=gt_data, scale=scale
            )
            
            if save_assets:
                base_filename = os.path.splitext(filename)[0]
                save_dir = os.path.join("assets", base_filename)
                os.makedirs(save_dir, exist_ok=True)
                for p_name, p_vis in panel_images.items():
                    cv2.imwrite(os.path.join(save_dir, f"{p_name}.jpg"), p_vis)

            if visualize:
                cv2.imshow(window_name, canvas)

            
        pred_print = f"{pred_vol:.1f} ml" if isinstance(pred_vol, (int, float)) else f"{silt_edge_y} px"
        print(f"[{idx+1}/{len(image_files)}] {filename} — Pred: {pred_print}{eval_text} | K={results['auto_k']}")
        
        if visualize:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n>> ออกจากโปรแกรมโดยผู้ใช้")
                break

    if visualize:
        cv2.destroyAllWindows()
    
    # บันทึกผลลัพธ์ลง CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        header = ['filename', 'water_top_y', 'water_bottom_y', 'silt_top_y', 'silt_base', 'pred_pixel', 'pred_volume']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"\nบันทึกผลลัพธ์ลงใน {output_file} สำเร็จ")

    if errors:
        mae = sum(errors) / len(errors)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print(f"  Total images evaluated: {len(errors)}")
        print(f"  Mean Absolute Error (Pixels): {mae:.2f} px")
        if vol_errors:
            mae_vol = sum(vol_errors) / len(vol_errors)
            print(f"  Mean Absolute Error (Volume): {mae_vol:.2f} ml")
        print("="*50)
    
    print("\nประมวลผลครบทุกรูปภาพแล้ว!")

if __name__ == "__main__":
    main(visualize=False, save_assets=True)

import cv2
import numpy as np
import os
import csv
from lib.silt_detector import SiltDetector

# --- CONFIGURATION PARAMETERS ---
DEFAULT_CONFIG = {
    # Pipeline Settings
    "data_dir": "data",
    "gt_file": "data/ground_truth.csv",
    "output_file": "detection_results.csv",
    "valid_ext": ('.jpg', '.jpeg', '.png'),
    "max_width": 600,
    "save_linear_comparison": True,
    
    # Algorithm Parameters
    "vol_exponent": 2.3, # ! Need (Cone shape exponent)
    "vol_exponent_linear": 1.0, # ! Need (Linear shape exponent)
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": (8, 8),
    "proj_window": 10, # ! Need
    "proj_threshold_ratio": 0.2, # ! Need
    "k_min": 3, # ! Need
    "k_max": 10, # ! Need
    "k_offset": 2, # ! Need
    "grabcut_iters": 5,
    "mask_threshold": 45,
    "y_threshold_ratio": 0.1, # ! Need
    "pos_weight_exponent": 2,
}
# ---------------------------------


def main(visualize=True, save_assets=True, config=None):
    if config is None:
        config = DEFAULT_CONFIG.copy()
        
    # Override visualize/save_assets if provided as arguments
    config['visualize'] = visualize
    config['save_assets'] = save_assets
    
    data_dir = config["data_dir"]
    gt_file = config["gt_file"]
    output_file = config["output_file"]
    valid_ext = config["valid_ext"]
    
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

    # แยก Config ส่วน Algorithm ให้ SiltDetector
    algo_params = {k: v for k, v in config.items() if k in [
        "vol_exponent", "clahe_clip_limit", "clahe_tile_size", "proj_window",
        "proj_threshold_ratio", "k_min", "k_max", "k_offset", "grabcut_iters",
        "mask_threshold", "y_threshold_ratio", "pos_weight_exponent"
    ]}
    detector = SiltDetector(**algo_params)

    if visualize:
        window_name = "Silt Detection Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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
        max_w = config["max_width"]
        if width > max_w:
            scale = max_w / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
            height, width = img.shape[:2] 
            
        # ประมวลผลภาพ
        results = detector.process(img)
        
        silt_edge_y = results['silt_edge_y']
        
        # คำนวณค่า Pred ใน Original Scale
        pred_orig = int(silt_edge_y / scale) if silt_edge_y is not None else ""
        
        pred_vol_cone = ""
        pred_vol_linear = ""
        if filename in ground_truth and pred_orig != "":
            gt_row = ground_truth[filename]
            # คำนวณตาม Exponent ที่ระบุใน Config
            pred_vol_cone = detector.estimate_volume(pred_orig, gt_row['water_top_y'], gt_row['water_bottom_y'], exponent=config['vol_exponent'])
            
            if config.get("save_linear_comparison", False):
                pred_vol_linear = detector.estimate_volume(pred_orig, gt_row['water_top_y'], gt_row['water_bottom_y'], exponent=config['vol_exponent_linear'])
            else:
                pred_vol_linear = ""
        
        row = {
            'filename': filename,
            'water_top_y': ground_truth[filename]['water_top_y'] if filename in ground_truth else "",
            'water_bottom_y': ground_truth[filename]['water_bottom_y'] if filename in ground_truth else "",
            'silt_top_y': ground_truth[filename]['silt_top_y'] if filename in ground_truth else "",
            'silt_base': ground_truth[filename]['silt_base'] if filename in ground_truth else "",
            'pred_pixel': pred_orig,
            'pred_vol_cone': pred_vol_cone,
            'pred_vol_linear': pred_vol_linear
        }
        results_data.append(row)

        eval_text = ""

        if visualize or save_assets:
            gt_data = ground_truth.get(filename)
            canvas, panel_images = detector.create_visualization(
                img, results, pred_vol=pred_vol_cone, gt_data=gt_data, scale=scale
            )
            
            if save_assets:
                base_filename = os.path.splitext(filename)[0]
                save_dir = os.path.join("assets", base_filename)
                os.makedirs(save_dir, exist_ok=True)
                for p_name, p_vis in panel_images.items():
                    cv2.imwrite(os.path.join(save_dir, f"{p_name}.jpg"), p_vis)

            if visualize:
                cv2.imshow(window_name, canvas)

            
        pred_print = f"{pred_vol_cone:.1f} ml (cone)" if isinstance(pred_vol_cone, (int, float)) else f"{silt_edge_y} px"
        print(f"[{idx+1}/{len(image_files)}] {filename} — Pred: {pred_print}{eval_text} | K={results['auto_k']}")
        
        if visualize:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n>> ออกจากโปรแกรมโดยผู้ใช้")
                break

    if visualize:
        cv2.destroyAllWindows()
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        header = ['filename', 'water_top_y', 'water_bottom_y', 'silt_top_y', 'silt_base', 'pred_pixel', 'pred_vol_cone']
        if config.get("save_linear_comparison", False):
            header.append('pred_vol_linear')
            
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        
        # กรองข้อมูลให้ตรงกับ header ก่อนเซฟ
        rows_to_save = []
        for r in results_data:
            rows_to_save.append({k: r[k] for k in header})
        writer.writerows(rows_to_save)
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
    main(visualize=False, save_assets=False)

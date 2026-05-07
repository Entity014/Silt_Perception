import cv2
import numpy as np
import os
import csv
from lib.silt_detector import SiltDetector

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    if dist == 0: return
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start_ptr = int(i * dash_length)
        end_ptr = int((i + 0.5) * dash_length)
        
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ptr / dist),
              int(pt1[1] + (pt2[1] - pt1[1]) * start_ptr / dist))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ptr / dist),
              int(pt1[1] + (pt2[1] - pt1[1]) * end_ptr / dist))
        cv2.line(img, p1, p2, color, thickness)

def main(visualize=True, save_assets=True):
    data_dir = "data"
    gt_file = os.path.join(data_dir, "ground_truth.csv")
    output_file = "detection_results.csv"
    valid_ext = ('.jpg', '.jpeg', '.png')
    
    # UI Constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2
    
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
                        'silt_top_y': int(row['silt_top_y'])
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
        
        # เตรียมข้อมูลสำหรับเซฟลง CSV (ให้เหมือน GT แต่เพิ่ม pred)
        row = {
            'filename': filename,
            'water_top_y': ground_truth[filename]['water_top_y'] if filename in ground_truth else "",
            'water_bottom_y': ground_truth[filename]['water_bottom_y'] if filename in ground_truth else "",
            'silt_top_y': ground_truth[filename]['silt_top_y'] if filename in ground_truth else "",
            'pred': pred_orig
        }
        results_data.append(row)

        # เปรียบเทียบกับ Ground Truth
        eval_text = ""
        display_img = balanced_img.copy()

        if filename in ground_truth:
            gt_y_orig = ground_truth[filename]['silt_top_y']
            gt_y = int(gt_y_orig * scale)
            
            if visualize or save_assets:
                # วาดเส้น Ground Truth (สีเขียว เส้นประ)
                draw_dashed_line(display_img, (lx, gt_y), (rx, gt_y), (0, 255, 0), thickness=2, dash_length=15)
                cv2.putText(display_img, f"GT: {gt_y}", (lx + 5, gt_y + 20), 
                            FONT, FONT_SCALE, (0, 255, 0), THICKNESS)
            
            if silt_edge_y is not None:
                error = abs(silt_edge_y - gt_y)
                errors.append(error)
                eval_text = f" | GT: {gt_y} | Error: {error}"
                if visualize or save_assets:
                    # วาด Error ไว้มุมขวาบน
                    cv2.putText(display_img, f"Error: {error} px", (width - 150, 30), 
                                FONT, FONT_SCALE, (0, 255, 255), THICKNESS)

        if visualize or save_assets:
            # ดึง intermediate images จาก results
            roi_img = results['roi_img']
            grabcut_img = results['grabcut_img']
            kmeans_img = results['kmeans_img']
            silt_mask = results['silt_mask']

            # --- CLAHE + เส้น lx/rx + Projection curve ---
            clahe_vis = balanced_img.copy()
            proj_smoothed = results['proj_smoothed']
            proj_max_val = results['proj_max_val']
            proj_threshold = results['proj_threshold']

            # วาดเส้น lx, rx
            cv2.line(clahe_vis, (lx, 0), (lx, height), (0, 0, 255), THICKNESS)
            cv2.line(clahe_vis, (rx, 0), (rx, height), (0, 0, 255), THICKNESS)
            cv2.putText(clahe_vis, f"lx={lx}", (lx+3, 20), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
            cv2.putText(clahe_vis, f"rx={rx}", (rx+3, 20), FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
            
            # วาด projection curve ลงบนรูปโดยตรง (ใช้พื้นที่ 1/3 ล่าง)
            if proj_max_val > 0:
                plot_h = height // 3
                y_base = height - 10
                
                # วาดเส้น threshold (แดง เส้นประ)
                # thresh_y = int(y_base - (proj_threshold / proj_max_val) * (plot_h - 20))
                # for x in range(0, width, 15):
                #     cv2.line(clahe_vis, (x, thresh_y), (min(x+8, width-1), thresh_y), (0, 0, 255), THICKNESS)
                
                # วาดกราฟ projection (เขียว)
                pts = []
                for x in range(width):
                    val = proj_smoothed[x] if x < len(proj_smoothed) else 0
                    y = int(y_base - (val / proj_max_val) * (plot_h - 20))
                    pts.append((x, y))
                pts = np.array(pts, dtype=np.int32)
                # alpha blend กราฟให้ดูเนียนขึ้น
                overlay = clahe_vis.copy()
                cv2.polylines(overlay, [pts], isClosed=False, color=(255, 255, 0), thickness=THICKNESS)
                cv2.addWeighted(overlay, 0.8, clahe_vis, 0.2, 0, clahe_vis)

            # --- ตกแต่ง Y-Histogram ---
            for i, (start_y, end_y) in enumerate(y_clusters):
                prob = cluster_probs[i]
                color = (0, 0, 255) if i == best_cluster_idx else (100, 100, 100)
                cv2.rectangle(y_hist_display, (0, start_y), (y_hist_display.shape[1]-1, end_y), color, THICKNESS)
                cv2.putText(y_hist_display, f"{prob:.2f}", (y_hist_display.shape[1]-100, start_y+15), 
                            FONT, FONT_SCALE, color, THICKNESS)
            if silt_edge_y is not None:
                cv2.line(y_hist_display, (0, silt_edge_y), (y_hist_display.shape[1], silt_edge_y), (255, 255, 0), 2)

            # --- วาดผลลัพธ์ (Pred/GT) บน display_img ---
            if silt_edge_y is not None:
                cv2.line(display_img, (lx, silt_edge_y), (rx, silt_edge_y), (0, 0, 255), 2)
                cv2.putText(display_img, f"Pred: {silt_edge_y}", (lx + 5, silt_edge_y - 10), 
                            FONT, FONT_SCALE, (0, 0, 255), THICKNESS)
            
            # === สร้างภาพ stack แนวนอน ===
            def label_img(image, text):
                bar_h = 25
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, text, (5, 18), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
                return np.vstack([bar, image])

            def resize_to_h(image, target_h):
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                h, w = image.shape[:2]
                new_w = max(1, int(w * target_h / h))
                return cv2.resize(image, (new_w, target_h))

            row_h = height

            panel_data = [
                (img, "1.Original"),
                (clahe_vis, "2.CLAHE + PROJECTION"),
                (roi_img, "3.ROI"),
                (grabcut_img, "4.GrabCut"),
                (kmeans_img, "5.K-Means"),
                (silt_mask, "6.Silt Mask"),
                (y_hist_display, "7.Y-Hist"),
                (display_img, "8.Result"),
            ]

            panels = []
            
            # สร้างโฟลเดอร์ใน assets ตามชื่อไฟล์
            base_filename = os.path.splitext(filename)[0]
            save_dir = os.path.join("assets", base_filename)
            os.makedirs(save_dir, exist_ok=True)
            
            for p_img, p_name in panel_data:
                panel_img = label_img(resize_to_h(p_img, row_h), p_name)
                panels.append(panel_img)
                # เซฟรูปลงในโฟลเดอร์ (ใช้ชื่อตาม label)
                cv2.imwrite(os.path.join(save_dir, f"{p_name}.jpg"), panel_img)

            canvas = np.hstack(panels)
            if visualize:
                cv2.imshow("Silt Detection Pipeline", canvas)
            
        print(f"[{idx+1}/{len(image_files)}] {filename} — Pred: {silt_edge_y}{eval_text} | K={results['auto_k']}")
        
        if visualize:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("\n>> ออกจากโปรแกรมโดยผู้ใช้")
                break

    if visualize:
        cv2.destroyAllWindows()
    
    # บันทึกผลลัพธ์ลง CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        header = ['filename', 'water_top_y', 'water_bottom_y', 'silt_top_y', 'pred']
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results_data)
    print(f"\nบันทึกผลลัพธ์ลงใน {output_file} สำเร็จ")

    if errors:
        mae = sum(errors) / len(errors)
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print(f"  Total images evaluated: {len(errors)}")
        print(f"  Mean Absolute Error (MAE): {mae:.2f} pixels")
        print("="*50)
    
    print("\nประมวลผลครบทุกรูปภาพแล้ว!")

if __name__ == "__main__":
    main(visualize=False, save_assets=True)

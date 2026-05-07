import cv2
import numpy as np
import os

class SiltDetector:
    # UI Constants
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    THICKNESS = 2

    def __init__(self):
        pass

    def estimate_volume(self, silt_y, water_top_y, water_bottom_y):
        """
        แปลงจาก pixel เป็นปริมาณ (ml) สำหรับกรวยอิมฮอฟฟ์
        """
        if water_bottom_y == water_top_y:
            return 0
        H = water_bottom_y - water_top_y
        h = water_bottom_y - silt_y
        if H <= 0: return 0
        # ใช้สูตร (h/H)^3 * 1000 + 100 ตามที่ผู้ใช้ระบุใน main.py
        vol = 1000.0 * (h / H)**2.3
        return round(vol, 2)

    @staticmethod
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

    def auto_light_balance(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        balanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return balanced_img

    def detect_tube_projection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx_abs = np.abs(sobelx)
        projection = np.sum(sobelx_abs, axis=0)
        
        window = 10
        smoothed = np.convolve(projection, np.ones(window)/window, mode='same')
        
        mid = len(smoothed) // 2
        peak_left = np.argmax(smoothed[:mid])
        peak_right = mid + np.argmax(smoothed[mid:])
        
        max_val = np.max(smoothed)
        threshold = max_val * 0.2
        
        left_x = peak_left
        for i in range(peak_left, -1, -1):
            if smoothed[i] < threshold:
                left_x = i
                break
                
        right_x = peak_right
        for i in range(peak_right, len(smoothed)):
            if smoothed[i] < threshold:
                right_x = i
                break
                
        left_x = max(0, left_x)
        right_x = min(img.shape[1]-1, right_x)
        
        return left_x, right_x, smoothed, max_val, threshold

    def segment_kmeans_auto(self, roi_img, min_k=3, max_k=12):
        """
        รัน K-Means และหาค่า K ที่เหมาะสมที่สุดโดยอัตโนมัติ (Elbow Method)
        """
        Z = roi_img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        sample_size = min(Z.shape[0], 5000)
        idx = np.random.choice(Z.shape[0], sample_size, replace=False)
        Z_sample = Z[idx]
        
        distortions = []
        ks = range(min_k, max_k + 1)
        for k in ks:
            ret, label, center = cv2.kmeans(Z_sample, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
            distortions.append(ret)
            
        if len(distortions) < 3:
            best_k = min_k
        else:
            coords = np.vstack((ks, distortions)).T
            p1 = coords[0]
            p2 = coords[-1]
            
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            line_unit_vec = line_vec / (line_len + 1e-6)
            
            vec_p1_to_all = coords - p1
            proj_len = np.dot(vec_p1_to_all, line_unit_vec)
            proj_vec = np.outer(proj_len, line_unit_vec)
            
            dist_to_line = np.linalg.norm(vec_p1_to_all - proj_vec, axis=1)
            best_k = ks[np.argmax(dist_to_line)]
        
        best_k = best_k + 2
        ret, label, center = cv2.kmeans(Z, best_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        
        gray_centers = cv2.cvtColor(center.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
        min_color = np.min(gray_centers)
        
        res = center[label.flatten()]
        result = res.reshape((roi_img.shape))
        return result, best_k, min_color

    def segment_grabcut(self, roi_img):
        mask = np.zeros(roi_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        h, w = roi_img.shape[:2]
        rect = (2, 2, w-4, h-4)
        
        try:
            cv2.grabCut(roi_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            result = roi_img.copy()
            result[mask2 == 0] = [255, 255, 255]
            return result
        except:
            return roi_img.copy()

    def compute_mask_y_histogram(self, mask):
        h, w = mask.shape
        profile = np.sum(mask, axis=1) / 255
        
        plot_w = 200
        plot_img = np.zeros((h, plot_w, 3), dtype=np.uint8)
        
        for i in range(0, plot_w, 50):
            cv2.line(plot_img, (i, 0), (i, h), (40, 40, 40), 1)
        
        points = []
        for y in range(h):
            x = int((profile[y] / w) * plot_w)
            x = min(max(x, 0), plot_w - 1)
            points.append((x, y))
            
        points = np.array(points, dtype=np.int32)
        cv2.polylines(plot_img, [points], isClosed=False, color=(0, 255, 0), thickness=2)
        
        return plot_img, profile

    def compute_color_histogram_image(self, roi_img, height_to_match):
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        max_val = np.max(hist)
        most_freq_color = np.argmax(hist)
        
        min_color = 0
        for i in range(256):
            if hist[i] > 0:
                min_color = i
                break
                
        hist_h, hist_w = 250, 256
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        
        if max_val > 0:
            hist_norm = (hist / max_val * (hist_h - 20)).astype(int)
            for x in range(256):
                cv2.line(hist_img, (x, hist_h), (x, hist_h - hist_norm[x][0]), (255, 255, 255), 1)
                
        cv2.line(hist_img, (most_freq_color, 0), (most_freq_color, hist_h), (0, 0, 255), 2)
        cv2.line(hist_img, (min_color, 0), (min_color, hist_h), (0, 255, 0), 2)
        
        cv2.putText(hist_img, f"Mode: {most_freq_color}", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(hist_img, f"Min: {min_color}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        hist_img_resized = cv2.resize(hist_img, (int(hist_w * height_to_match / hist_h), height_to_match))
        
        return hist_img_resized, most_freq_color, min_color

    def get_silt_mask(self, roi_img, target_gray=None):
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        if target_gray is not None:
            t_gray = int(target_gray)
            mask = cv2.inRange(gray, t_gray, t_gray)
        else:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)
        
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return mask

    def find_silt_edge_probabilistic(self, y_profile, roi_h):
        y_max = np.max(y_profile)
        y_threshold = y_max * 0.1
        y_clusters = []
        c_start = None
        
        for y in range(len(y_profile)):
            if y_profile[y] > y_threshold:
                if c_start is None: c_start = y
            else:
                if c_start is not None:
                    y_clusters.append((c_start, y-1))
                    c_start = None
        if c_start is not None: y_clusters.append((c_start, len(y_profile)-1))
        
        scores = []
        for start_y, end_y in y_clusters:
            cluster_area = np.sum(y_profile[start_y:end_y+1])
            center_y = (start_y + end_y) / 2
            pos_weight = (center_y / roi_h) ** 2
            scores.append(cluster_area * pos_weight)
            
        total_score = sum(scores)
        if total_score > 0:
            cluster_probs = [s / total_score for s in scores]
        else:
            cluster_probs = [1.0/len(y_clusters)] * len(y_clusters) if y_clusters else []

        best_cluster_idx = np.argmax(cluster_probs) if cluster_probs else -1
        silt_edge_y = None
        
        if best_cluster_idx != -1:
            silt_edge_y = y_clusters[best_cluster_idx][0]
            
        return silt_edge_y, y_clusters, cluster_probs, best_cluster_idx

    def process(self, img):
        """
        Run the full pipeline on a single image and return results.
        """
        height, width = img.shape[:2]
        
        # 1. CLAHE
        balanced_img = self.auto_light_balance(img)
            
        # 2. Tube Projection
        lx, rx, smoothed, max_val, threshold = self.detect_tube_projection(img)
        if rx - lx < 50:
            lx, rx = 0, width
            
        # 3. ROI
        roi_img = img[:, lx:rx].copy()
        
        # 4. GrabCut
        grabcut_img = self.segment_grabcut(roi_img)
        
        # 5. K-Means
        kmeans_img, auto_k, kmeans_min = self.segment_kmeans_auto(grabcut_img, min_k=3, max_k=10)
        kmeans_img = cv2.medianBlur(kmeans_img, 5)
        
        # 6. Histograms
        kmeans_h = kmeans_img.shape[0]
        color_hist_display, dominant_gray, hist_min = self.compute_color_histogram_image(kmeans_img, kmeans_h)
        
        # 7. Silt Mask
        silt_mask = self.get_silt_mask(kmeans_img, target_gray=kmeans_min)
        
        # 8. Y-Histogram & Silt Edge
        y_hist_display, y_profile = self.compute_mask_y_histogram(silt_mask)
        roi_h = roi_img.shape[0]
        silt_edge_y, y_clusters, cluster_probs, best_cluster_idx = self.find_silt_edge_probabilistic(y_profile, roi_h)
        
        return {
            'balanced_img': balanced_img,
            'proj_smoothed': smoothed,
            'proj_max_val': max_val,
            'proj_threshold': threshold,
            'roi_img': roi_img,
            'grabcut_img': grabcut_img,
            'kmeans_img': kmeans_img,
            'silt_mask': silt_mask,
            'lx': lx,
            'rx': rx,
            'silt_edge_y': silt_edge_y,
            'auto_k': auto_k,
            'kmeans_min': kmeans_min,
            'y_hist_display': y_hist_display,
            'y_clusters': y_clusters,
            'cluster_probs': cluster_probs,
            'best_cluster_idx': best_cluster_idx
        }

    def label_img(self, image, text):
        bar_h = 25
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        bar = np.zeros((bar_h, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (5, 18), self.FONT, self.FONT_SCALE, (255, 255, 255), self.THICKNESS)
        return np.vstack([bar, image])

    def resize_to_h(self, image, target_h):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        new_w = max(1, int(w * target_h / h))
        return cv2.resize(image, (new_w, target_h))

    def create_visualization(self, original_img, results, pred_vol=None, gt_data=None, scale=1.0):
        """
        สร้างภาพ Visualization รวมทุกขั้นตอน
        """
        height, width = original_img.shape[:2]
        lx, rx = results['lx'], results['rx']
        silt_edge_y = results['silt_edge_y']
        balanced_img = results['balanced_img']
        y_hist_display = results['y_hist_display'].copy()
        
        # --- เตรียม Result Image (Panel 8) ---
        display_img = balanced_img.copy()
        
        # วาด Pred
        if silt_edge_y is not None:
            cv2.line(display_img, (lx, silt_edge_y), (rx, silt_edge_y), (0, 0, 255), 2)
            pred_text = f"Pred: {pred_vol:.1f} ml" if isinstance(pred_vol, (int, float)) else f"Pred: {silt_edge_y} px"
            cv2.putText(display_img, pred_text, (lx + 5, silt_edge_y - 10), 
                        self.FONT, self.FONT_SCALE, (0, 0, 255), self.THICKNESS)

        # วาด GT
        if gt_data:
            gt_silt_y = int(gt_data['silt_top_y'] * scale)
            gt_water_top_y = int(gt_data['water_top_y'] * scale)
            gt_water_bottom_y = int(gt_data['water_bottom_y'] * scale)
            
            # คำนวณ GT Vol สำหรับวาด
            gt_vol_vis = self.estimate_volume(gt_silt_y, gt_water_top_y, gt_water_bottom_y)



        # --- ตกแต่ง CLAHE + Projection (Panel 2) ---
        clahe_vis = balanced_img.copy()
        proj_smoothed = results['proj_smoothed']
        proj_max_val = results['proj_max_val']
        
        cv2.line(clahe_vis, (lx, 0), (lx, height), (0, 0, 255), self.THICKNESS)
        cv2.line(clahe_vis, (rx, 0), (rx, height), (0, 0, 255), self.THICKNESS)
        
        if proj_max_val > 0:
            plot_h = height // 3
            y_base = height - 10
            pts = []
            for x in range(width):
                val = proj_smoothed[x] if x < len(proj_smoothed) else 0
                y = int(y_base - (val / proj_max_val) * (plot_h - 20))
                pts.append((x, y))
            pts = np.array(pts, dtype=np.int32)
            overlay = clahe_vis.copy()
            cv2.polylines(overlay, [pts], isClosed=False, color=(255, 255, 0), thickness=self.THICKNESS)
            cv2.addWeighted(overlay, 0.8, clahe_vis, 0.2, 0, clahe_vis)

        # --- ตกแต่ง Y-Histogram (Panel 7) ---
        y_clusters = results['y_clusters']
        cluster_probs = results['cluster_probs']
        best_cluster_idx = results['best_cluster_idx']
        for i, (start_y, end_y) in enumerate(y_clusters):
            prob = cluster_probs[i]
            color = (0, 0, 255) if i == best_cluster_idx else (100, 100, 100)
            cv2.rectangle(y_hist_display, (0, start_y), (y_hist_display.shape[1]-1, end_y), color, self.THICKNESS)
            cv2.putText(y_hist_display, f"{prob:.2f}", (y_hist_display.shape[1]-100, start_y+15), 
                        self.FONT, self.FONT_SCALE, color, self.THICKNESS)
        if silt_edge_y is not None:
            cv2.line(y_hist_display, (0, silt_edge_y), (y_hist_display.shape[1], silt_edge_y), (255, 255, 0), 2)

        # --- รวมพาเนล ---
        panel_data = [
            (original_img, "1.Original"),
            (clahe_vis, "2.CLAHE + PROJECTION"),
            (results['roi_img'], "3.ROI"),
            (results['grabcut_img'], "4.GrabCut"),
            (results['kmeans_img'], "5.K-Means"),
            (results['silt_mask'], "6.Silt Mask"),
            (y_hist_display, "7.Y-Hist"),
            (display_img, "8.Result"),
        ]

        panels = []
        panel_images = {} # สำหรับเซฟแยกไฟล์
        for p_img, p_name in panel_data:
            p_vis = self.label_img(self.resize_to_h(p_img, height), p_name)
            panels.append(p_vis)
            panel_images[p_name] = p_vis

        canvas = np.hstack(panels)
        return canvas, panel_images

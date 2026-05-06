import cv2
import numpy as np
import os

class SiltDetector:
    def __init__(self):
        pass

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

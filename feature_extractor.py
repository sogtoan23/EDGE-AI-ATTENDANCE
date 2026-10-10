import os
import cv2
import numpy as np
import onnxruntime as ort

class FullPipelineFaceExtractor:
    def __init__(
        self,
        det_model_path="models/det_500m.onnx",
        spoof_model_path="models/minifasnet.onnx",
        rec_model_path="models/w600k_mbf.onnx"
    ):
        for p in [det_model_path, spoof_model_path, rec_model_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Không tìm thấy file model: {p}")

        # Kiểm tra phần cứng: Ưu tiên GPU nếu có, mặc định CPU
        available_providers = ort.get_available_providers()
        preferred_providers = []
        if 'CUDAExecutionProvider' in available_providers:
            preferred_providers.append('CUDAExecutionProvider')
        if 'TensorrtExecutionProvider' in available_providers:
            preferred_providers.append('TensorrtExecutionProvider')
        preferred_providers.append('CPUExecutionProvider')

        # 1. Model Detection (det_500m.onnx)
        self.det_session = ort.InferenceSession(det_model_path, providers=preferred_providers)
        self.det_input_name = self.det_session.get_inputs()[0].name
        
        # 2. Model Anti-Spoofing / Liveness (minifasnet.onnx)
        self.spoof_session = ort.InferenceSession(spoof_model_path, providers=preferred_providers)
        self.spoof_input_name = self.spoof_session.get_inputs()[0].name
        self.spoof_output_name = self.spoof_session.get_outputs()[0].name

        # 3. Model Feature Extraction / ArcFace (w600k_mbf.onnx)
        self.rec_session = ort.InferenceSession(rec_model_path, providers=preferred_providers)
        self.rec_input_name = self.rec_session.get_inputs()[0].name
        self.rec_output_name = self.rec_session.get_outputs()[0].name

        # Mẫu landmark 5 điểm chuẩn hóa ArcFace 112x112
        self.ref_landmarks = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    # -------------------------------------------------------------
    # MODEL 1: DETECT KHUÔN MẶT & LANDMARKS 5 ĐIỂM (SCRFD)
    # -------------------------------------------------------------
    def detect_face(self, img, conf_thresh=0.5):
        h, w = img.shape[:2]
        input_size = (640, 640)
        im_ratio = float(h) / w
        model_ratio = float(input_size[1]) / input_size[0]
        
        if im_ratio > model_ratio:
            new_h = input_size[1]
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_size[0]
            new_h = int(new_w * im_ratio)
            
        resized_img = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized_img
        
        det_tensor = (det_img.astype(np.float32) - 127.5) / 128.0
        det_tensor = np.transpose(det_tensor, (2, 0, 1))
        det_tensor = np.expand_dims(det_tensor, axis=0)

        outputs = self.det_session.run(None, {self.det_input_name: det_tensor})
        strides = [8, 16, 32]
        num_strides = len(strides)
        scores_list, bboxes_list, kpss_list = [], [], []

        for idx, stride in enumerate(strides):
            scores = outputs[idx]
            bbox_preds = outputs[idx + num_strides] * stride
            kps_preds = outputs[idx + num_strides * 2] * stride
            
            height = input_size[1] // stride
            width = input_size[0] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32) * stride
            anchor_centers = np.repeat(anchor_centers.reshape(-1, 2), 2, axis=0)

            pos_inds = np.where(scores.reshape(-1, 1) >= conf_thresh)[0]
            if len(pos_inds) > 0:
                bboxes = np.zeros((len(pos_inds), 4), dtype=np.float32)
                cx = anchor_centers[pos_inds, 0]
                cy = anchor_centers[pos_inds, 1]
                
                bboxes[:, 0] = cx - bbox_preds.reshape(-1, 4)[pos_inds, 0]
                bboxes[:, 1] = cy - bbox_preds.reshape(-1, 4)[pos_inds, 1]
                bboxes[:, 2] = cx + bbox_preds.reshape(-1, 4)[pos_inds, 2]
                bboxes[:, 3] = cy + bbox_preds.reshape(-1, 4)[pos_inds, 3]

                kpss = kps_preds.reshape(-1, 5, 2)[pos_inds]
                kpss[:, :, 0] += cx[:, None]
                kpss[:, :, 1] += cy[:, None]

                scores_list.append(scores.reshape(-1)[pos_inds])
                bboxes_list.append(bboxes)
                kpss_list.append(kpss)

        if not scores_list:
            return None, None, None

        scores = np.concatenate(scores_list)
        bboxes = np.concatenate(bboxes_list)
        kpss = np.concatenate(kpss_list)

        scale_x = float(w) / new_w
        scale_y = float(h) / new_h
        bboxes[:, [0, 2]] *= scale_x
        bboxes[:, [1, 3]] *= scale_y
        kpss[:, :, 0] *= scale_x
        kpss[:, :, 1] *= scale_y

        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        best_idx = np.argmax(areas)
        return bboxes[best_idx], kpss[best_idx], scores[best_idx]

    # -------------------------------------------------------------
    # MODEL 2: CHỐNG GIẢ MẠO / LIVENESS (MINIFASNET)
    # Paper quy ước: [Index 0: Paper, Index 1: Real, Index 2: Screen]
    # -------------------------------------------------------------
    def check_liveness(self, img, bbox):
        x1, y1, x2, y2 = bbox.astype(int)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        # Scale chuẩn 2.7x của Silent-Face-Anti-Spoofing
        scale = 2.7
        new_w, new_h = int(w * scale), int(h * scale)
        nx1 = max(0, cx - new_w // 2)
        ny1 = max(0, cy - new_h // 2)
        nx2 = min(img.shape[1], cx + new_w // 2)
        ny2 = min(img.shape[0], cy + new_h // 2)
        
        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return 0.0

        crop = cv2.resize(crop, (80, 80)).astype(np.float32)
        crop = np.transpose(crop, (2, 0, 1))
        crop = np.expand_dims(crop, axis=0)

        outputs = self.spoof_session.run([self.spoof_output_name], {self.spoof_input_name: crop})[0]
        
        # Hàm xác suất Softmax
        exp_vals = np.exp(outputs[0] - np.max(outputs[0]))
        probs = exp_vals / np.sum(exp_vals)

        # Index 1 luôn là REAL FACE
        real_score = float(probs[1])

        print(f"[DEBUG LIVENESS] Probs: [Paper={probs[0]:.4f}, Real={probs[1]:.4f}, Screen={probs[2]:.4f}] | Real Score: {real_score:.4f}")
        return real_score

    # -------------------------------------------------------------
    # ALIGN KHUÔN MẶT THEO 5 ĐIỂM LANDMARKS
    # -------------------------------------------------------------
    def align_face(self, img, landmarks):
        M, _ = cv2.estimateAffinePartial2D(landmarks, self.ref_landmarks)
        if M is None:
            return cv2.resize(img, (112, 112))
        return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

    # -------------------------------------------------------------
    # MODEL 3: TRÍCH XUẤT ĐẶC TRƯNG VECTOR 512-D (ARCFACE / MBF)
    # -------------------------------------------------------------
    def extract_features(self, aligned_face):
        face_img = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = np.expand_dims(face_img, axis=0)

        embedding = self.rec_session.run([self.rec_output_name], {self.rec_input_name: face_img})[0][0]
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    # -------------------------------------------------------------
    # PIPELINE HOÀN CHỈNH BẮT BUỘC 3 MODEL TUẦN TỰ
    # -------------------------------------------------------------
    def process_multiple_photos(self, photos_bytes: list, liveness_thresh=0.70):
        embeddings = []
        liveness_scores = []

        for idx, p_bytes in enumerate(photos_bytes):
            nparr = np.frombuffer(p_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Góc ảnh {idx + 1}: Dữ liệu ảnh không hợp lệ.")

            # BƯỚC 1: Model 1 phát hiện khuôn mặt
            bbox, landmarks, _ = self.detect_face(img)
            if bbox is None:
                raise ValueError(f"Góc ảnh {idx + 1}: Không phát hiện được khuôn mặt rõ ràng trong khung hình.")

            # BƯỚC 2: Model 2 thẩm định mặt thật hay giả
            liveness = self.check_liveness(img, bbox)
            if liveness < liveness_thresh:
                # Dừng toàn bộ chu trình ngay lập tức, không gọi sang Model 3
                raise ValueError(
                    f"Cảnh báo giả mạo! Góc ảnh {idx + 1} bị từ chối do phát hiện màn hình thiết bị / ảnh in (Độ tin cậy mặt thật: {liveness*100:.1f}%)."
                )

            liveness_scores.append(liveness)

            # BƯỚC 3: Chỉ khi là mặt thật mới align và trích xuất vector 512-D
            aligned = self.align_face(img, landmarks)
            emb = self.extract_features(aligned)
            embeddings.append(emb)

        if len(embeddings) == 0:
            raise ValueError("Không phát hiện khuôn mặt hợp lệ trong các góc chụp.")

        mean_embedding = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm

        avg_liveness = sum(liveness_scores) / len(liveness_scores)
        return mean_embedding.tolist(), avg_liveness
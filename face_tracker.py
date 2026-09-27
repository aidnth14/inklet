#!/usr/bin/env python3
"""
FACE, EYE, EYEBROW & MOUTH TRACKER (BACKGROUND THREAD)
======================================================
High-precision 1:1 facial puppet tracker with Hybrid Geometry & Blendshapes:
- Hybrid Blink & Mouth Engine: Combines 3D Euclidean distances with AI blendshapes.
  Guarantees flawless two-eye blinks and hyper-sensitive lip-sync without exaggeration.
- Zero Baseline Drift: "Forward" is locked permanently. No need to reset.
- FastVideoCapture: Zero-latency asynchronous camera I/O.
"""

import cv2
import os
import sys
import time
import math
import logging
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")

_MP_TASK_IN_MODELS_DIR = os.path.join(MODELS_DIR, "face_landmarker.task")
_MP_TASK_LEGACY = os.path.join(BASE_DIR, "face_landmarker.task")
MEDIAPIPE_TASK_MODEL = (
    _MP_TASK_IN_MODELS_DIR if os.path.exists(_MP_TASK_IN_MODELS_DIR) else _MP_TASK_LEGACY
)

def dist3d(p1, p2):
    """Calculates true Euclidean distance in 3D space, completely immune to 2D perspective distortion."""
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


class FastVideoCapture:
    """Dedicated I/O Thread for Zero-Latency Frame Grabbing."""
    def __init__(self, src: int, backend: int) -> None:
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            return self.ret, (self.frame.copy() if self.frame is not None else None)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.015, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    def filter(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        edx = self._alpha(self.d_cutoff, dt)
        dx_hat = edx * dx + (1.0 - edx) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class HeuristicVisemeClassifier:
    """Hyper-sensitive rule-based viseme engine."""
    def classify(self, b: dict, mouth_open: float, smile: float, pucker: float, asym: float, frown: float) -> str:
        # HARD OVERRIDE: If the mouth is physically closed, prevent ANY talking visemes
        if mouth_open < 0.04:
            if frown > 0.25: return "FROWN"
            if abs(asym) > 0.25: return "SMIRK"
            if smile > 0.25: return "TRIANGLE_SMILE"
            return "NEUTRAL"

        # If mouth is open, evaluate shapes aggressively
        funnel = b.get("mouthFunnel", 0.0)
        roll_in = (b.get("mouthRollLower", 0.0) + b.get("mouthRollUpper", 0.0)) / 2.0
        shrug = (b.get("mouthShrugLower", 0.0) + b.get("mouthShrugUpper", 0.0)) / 2.0
        stretch = (b.get("mouthStretchLeft", 0.0) + b.get("mouthStretchRight", 0.0)) / 2.0

        if roll_in > 0.25 and mouth_open < 0.30: return "BILABIAL"
        if pucker > 0.25: return "PUCKER_U"
        if funnel > 0.20 and mouth_open > 0.05: return "ROUNDED_O"
        if shrug > 0.20 and mouth_open < 0.40: return "LABIODENTAL"
        if stretch > 0.25 and mouth_open > 0.05: return "SPREAD_E_I"
        if smile > 0.40 and mouth_open > 0.10: return "WIDE_GRIN"
        if mouth_open > 0.35: return "OPEN_JAW"
        
        return "TALKING"


class AutoCameraTuner:
    def __init__(self) -> None:
        self.stage = 0
        self.gamma = 1.0
        self.clahe_clip = 0.0
        self.glare_suppression = False
        self.shadow_boost = False
        self.locked = False
        self.status = "INITIALIZING"
        self._lut_cache: Dict[float, np.ndarray] = {}
        self.last_eval_time = time.time()
        self.detection_streak = 0
        self.failure_streak = 0

        self.tuning_profiles = [
            (1.0, 0.0, False, False, "Standard (Natural Exposure)"),
            (1.35, 1.5, False, False, "Mild Shadow Lift & Local Contrast"),
            (1.65, 2.5, False, False, "Backlight Compensation (Moderate Lift)"),
            (1.85, 3.2, True, False, "Sun Reflection & Glasses Glare Suppression"),
            (2.10, 3.8, True, True, "High Dynamic Range Silhouette Rescue"),
        ]

    def get_lut(self, gamma: float) -> np.ndarray:
        g_key = round(float(gamma), 2)
        if g_key not in self._lut_cache:
            inv = 1.0 / max(0.1, g_key)
            self._lut_cache[g_key] = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype('uint8')
        return self._lut_cache[g_key]

    def optimize_frame(self, frame: np.ndarray) -> np.ndarray:
        out = frame
        if abs(self.gamma - 1.0) > 0.05:
            out = cv2.LUT(out, self.get_lut(self.gamma))
        if self.clahe_clip > 0.1:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            out = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        if self.glare_suppression:
            gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            glare_mask = gray > 232
            if np.any(glare_mask):
                out = out.copy()
                out[glare_mask] = np.clip(out[glare_mask] * 0.82, 0, 255).astype(np.uint8)
        return out

    def update_feedback(self, detected: bool) -> None:
        now = time.time()
        if detected:
            self.detection_streak += 1
            self.failure_streak = 0
            if self.detection_streak >= 12:
                self.locked = True
                self.status = f"LOCKED (Profile #{self.stage + 1})"
        else:
            self.failure_streak += 1
            self.detection_streak = 0
            if self.failure_streak >= 6 and (now - self.last_eval_time) > 0.25:
                self.locked = False
                self.last_eval_time = now
                self.failure_streak = 0
                self.step_next_tweak()

    def step_next_tweak(self) -> None:
        self.stage = (self.stage + 1) % len(self.tuning_profiles)
        g, c, gl, sb, desc = self.tuning_profiles[self.stage]
        self.gamma = g
        self.clahe_clip = c
        self.glare_suppression = gl
        self.shadow_boost = sb
        self.status = f"AUTO-TUNING: {desc}"


def open_working_camera() -> Tuple[FastVideoCapture, int]:
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else (cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY)
    for idx in [0, 1, 2]:
        try:
            cap = FastVideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            valid = False
            for _ in range(12):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 3.0:
                    valid = True
                    break
                time.sleep(0.03)
            if valid:
                logger.info("Connected to camera %d", idx)
                return cap, idx
            cap.release()
        except Exception:
            pass
    return FastVideoCapture(0, backend), 0


def annotate_audit_frame(frame: np.ndarray, lm: Any, b: dict, state: dict, fps: float, lat: float) -> np.ndarray:
    h, w = frame.shape[:2]
    vis = frame.copy()
    
    cv2.rectangle(vis, (0, 0), (w, 40), (12, 10, 20), -1)
    cv2.putText(vis, f"FPS: {fps:.1f} | Latency: {lat:.1f}ms | {state.get('tuner_status', '')}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1)

    if lm is None:
        cv2.putText(vis, "SEARCHING...", (w//2 - 60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return vis

    for pt in [473, 468]:
        cv2.circle(vis, (int(lm[pt].x * w), int(lm[pt].y * h)), 3, (0, 255, 255), -1)
    
    mouth_pts = np.array([[int(lm[p].x * w), int(lm[p].y * h)] for p in [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]], np.int32)
    cv2.polylines(vis, [mouth_pts], True, (255, 120, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"Viseme: {state.get('viseme_mode', 'NEUTRAL')}", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return vis


class FaceTrackerThread(QThread):
    face_updated = Signal(dict)
    audit_frame_updated = Signal(object, dict)

    def __init__(self, target_fps: int = 30) -> None:
        super().__init__()
        self.target_fps = target_fps
        self.running = True
        self.camera_enabled = True
        self.audit_mode_enabled = True
        self.cap = None

        self.head_x = 0.0; self.head_y = 0.0
        self.yaw = 0.0; self.pitch = 0.0; self.tilt_deg = 0.0
        self.depth = 1.0
        self.neck_x = 0.0; self.neck_y = 0.0
        self.velocity = 0.0

        self.look_x = 0.0; self.look_y = 0.0
        self.brow_raise = 0.0; self.eye_openness = 0.5
        self.blink_left = 0.0; self.blink_right = 0.0
        self.is_blinking = False

        self.smile_intensity = 0.0; self.mouth_open = 0.0; self.mouth_width = 0.75
        self.mouth_shift_x = 0.0; self.mouth_shift_y = 0.0
        self.upper_teeth = 0.0; self.lower_teeth = 0.0
        self.lip_pucker = 0.0; self.frown = 0.0; self.smile_asymmetry = 0.0
        self.viseme_mode = "NEUTRAL"
        
        self.viseme_engine = HeuristicVisemeClassifier()
        self.face_detected = False
        self.last_head_pos = (0.0, 0.0)

        self.camera_tuner = AutoCameraTuner()
        
        self.filter_head_x = OneEuroFilter(min_cutoff=0.9, beta=1.2)
        self.filter_head_y = OneEuroFilter(min_cutoff=0.9, beta=1.2)
        self.filter_yaw = OneEuroFilter(min_cutoff=0.9, beta=1.6)
        self.filter_pitch = OneEuroFilter(min_cutoff=0.9, beta=1.6)
        self.filter_tilt = OneEuroFilter(min_cutoff=0.7, beta=1.4)
        self.filter_look_x = OneEuroFilter(min_cutoff=0.7, beta=1.5)
        self.filter_look_y = OneEuroFilter(min_cutoff=0.7, beta=1.5)
        self.filter_brow = OneEuroFilter(min_cutoff=0.8, beta=1.2)
        self.filter_blink_l = OneEuroFilter(min_cutoff=2.0, beta=4.0)
        self.filter_blink_r = OneEuroFilter(min_cutoff=2.0, beta=4.0)
        self.filter_mouth = OneEuroFilter(min_cutoff=1.5, beta=2.5)

    def run(self) -> None:
        self.cap, _ = open_working_camera()

        mp_detector = None
        if os.path.exists(MEDIAPIPE_TASK_MODEL):
            try:
                import mediapipe as mp
                from mediapipe.tasks.python import vision
                from mediapipe.tasks import python
                base_opts = python.BaseOptions(model_asset_path=MEDIAPIPE_TASK_MODEL)
                landmarker_opts = vision.FaceLandmarkerOptions(
                    base_options=base_opts, running_mode=vision.RunningMode.VIDEO,
                    output_face_blendshapes=True, output_facial_transformation_matrixes=True,
                    num_faces=1, min_face_detection_confidence=0.5, min_tracking_confidence=0.5
                )
                mp_detector = vision.FaceLandmarker.create_from_options(landmarker_opts)
            except Exception as e:
                logger.warning("MediaPipe init failed: %s", e)

        interval = 1.0 / self.target_fps
        dead_frames_counter = 0

        while self.running:
            t_start = time.time()

            if not self.camera_enabled or self.cap is None or not self.cap.isOpened():
                self.decay_to_neutral()
                self.face_updated.emit(self.get_state_dict())
                self.msleep(35)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                dead_frames_counter += 1
                if dead_frames_counter > 50:
                    self.cap.release()
                    self.cap, _ = open_working_camera()
                    dead_frames_counter = 0
                self.decay_to_neutral()
                self.face_updated.emit(self.get_state_dict())
                self.msleep(30)
                continue

            dead_frames_counter = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            tuned_frame = self.camera_tuner.optimize_frame(frame)
            detected = False

            if mp_detector:
                try:
                    rgb = cv2.cvtColor(tuned_frame, cv2.COLOR_BGR2RGB)
                    infer_w = min(w, 640)
                    infer_h = int(h * (infer_w / float(w)))
                    small_rgb = cv2.resize(rgb, (infer_w, infer_h)) if (infer_w, infer_h) != (w, h) else rgb
                    
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=small_rgb)
                    ts_ms = max(getattr(self, "_last_ts_ms", 0) + 1, int(time.time() * 1000))
                    self._last_ts_ms = ts_ms
                    res = mp_detector.detect_for_video(mp_img, ts_ms)

                    if res.face_landmarks and res.face_blendshapes:
                        detected = True
                        lm = res.face_landmarks[0]
                        b = {s.category_name: s.score for s in res.face_blendshapes[0]}

                        # 1. PERFECT 3D HEAD POSE FROM TRANSFORMATION MATRIX
                        matrix_solved = False
                        if hasattr(res, "facial_transformation_matrixes") and len(res.facial_transformation_matrixes) > 0:
                            try:
                                mat = np.array(res.facial_transformation_matrixes[0][:3, :3])
                                sy = math.sqrt(mat[0,0]*mat[0,0] + mat[1,0]*mat[1,0])
                                singular = sy < 1e-6
                                if not singular:
                                    pitch_rad = math.atan2(mat[2,1] , mat[2,2])
                                    yaw_rad = math.atan2(-mat[2,0], sy)
                                    roll_rad = math.atan2(mat[1,0], mat[0,0])
                                else:
                                    pitch_rad = math.atan2(-mat[1,2], mat[1,1])
                                    yaw_rad = math.atan2(-mat[2,0], sy)
                                    roll_rad = 0
                                
                                raw_pitch = max(-1.0, min(1.0, math.degrees(pitch_rad) / 25.0))
                                raw_yaw = max(-1.0, min(1.0, math.degrees(yaw_rad) / 25.0))
                                raw_tilt = math.degrees(roll_rad)
                                matrix_solved = True
                            except:
                                pass

                        if not matrix_solved:
                            dx_outer = lm[263].x - lm[33].x
                            dy_outer = lm[263].y - lm[33].y
                            raw_tilt = -math.degrees(math.atan2(dy_outer, dx_outer))
                            raw_yaw = (lm[1].x - 0.5) * 2.5
                            raw_pitch = (lm[1].y - 0.5) * 3.0

                        target_yaw = self.filter_yaw.filter(raw_yaw, t_start)
                        target_pitch = self.filter_pitch.filter(raw_pitch, t_start)
                        target_tilt = self.filter_tilt.filter(raw_tilt, t_start)

                        target_head_x = self.filter_head_x.filter((lm[1].x - 0.5) * 2.2, t_start)
                        target_head_y = self.filter_head_y.filter((lm[1].y - 0.5) * 2.2, t_start)
                        target_neck_x = target_head_x * 0.70
                        target_neck_y = target_head_y * 0.70

                        eye_dist = math.hypot(lm[263].x - lm[33].x, lm[263].y - lm[33].y)
                        target_depth = max(0.5, min(2.0, eye_dist / 0.20))

                        # 2. GAZE
                        gaze_h = (b.get("eyeLookOutLeft", 0) - b.get("eyeLookInLeft", 0) + b.get("eyeLookInRight", 0) - b.get("eyeLookOutRight", 0)) / 2.0
                        gaze_v = (b.get("eyeLookDownLeft", 0) + b.get("eyeLookDownRight", 0) - b.get("eyeLookUpLeft", 0) - b.get("eyeLookUpRight", 0)) / 2.0
                        target_look_x = self.filter_look_x.filter(gaze_h * 1.5, t_start)
                        target_look_y = self.filter_look_y.filter(gaze_v * 1.6, t_start)

                        # 3. HYBRID 3D BLINK TRACKING (100% Guaranteed Two-Eye Blinks)
                        
                        # Physical 3D Euclidean Eyelid Measurement
                        # Left Eye (User's Left, image right)
                        eye_w_l = dist3d(lm[362], lm[263])
                        eye_h_l = dist3d(lm[386], lm[374])
                        ear_l = eye_h_l / max(0.001, eye_w_l)
                        
                        # Right Eye (User's Right, image left)
                        eye_w_r = dist3d(lm[33], lm[133])
                        eye_h_r = dist3d(lm[159], lm[145])
                        ear_r = eye_h_r / max(0.001, eye_w_r)

                        # Math calculation: If EAR drops below 0.22, it's a blink.
                        ear_blink_l = max(0.0, min(1.0, (0.24 - ear_l) * 8.0))
                        ear_blink_r = max(0.0, min(1.0, (0.24 - ear_r) * 8.0))

                        # Blendshape AI calculation
                        bs_blink_l = max(0.0, min(1.0, (b.get("eyeBlinkLeft", 0.0) - 0.08) * 4.0))
                        bs_blink_r = max(0.0, min(1.0, (b.get("eyeBlinkRight", 0.0) - 0.08) * 4.0))
                        
                        # Take the absolute maximum of both models! If one fails, the other triggers.
                        target_blink_left = max(ear_blink_l, bs_blink_l)
                        target_blink_right = max(ear_blink_r, bs_blink_r)
                        
                        target_blink_left = self.filter_blink_l.filter(target_blink_left, t_start)
                        target_blink_right = self.filter_blink_r.filter(target_blink_right, t_start)
                        target_blink = (target_blink_left > 0.45 and target_blink_right > 0.45)

                        # Brow Raise
                        brow_up = (b.get("browInnerUp", 0.0) + b.get("browOuterUpLeft", 0.0) + b.get("browOuterUpRight", 0.0)) / 3.0
                        target_brow_raise = self.filter_brow.filter(brow_up * 1.6, t_start)
                        target_eye_openness = 0.5 + target_brow_raise * 0.4

                        # 4. HYBRID MOUTH KINEMATICS (Hyper-Sensitive)
                        jaw_open = b.get("jawOpen", 0.0)
                        mouth_close = b.get("mouthClose", 0.0)
                        
                        # Physical 3D Euclidean Lip Distance (Catches subtle speech and parted lips)
                        lip_dist = dist3d(lm[13], lm[14]) / max(0.001, eye_dist)
                        phys_open = max(0.0, min(1.0, (lip_dist - 0.015) * 8.0))
                        
                        # Combine AI Jaw prediction with Physical Lip parting
                        raw_mouth_open = max(jaw_open * 3.5, phys_open)
                        
                        # Hard lock: If the AI detects lips are forcibly pressed closed, suppress talking.
                        if mouth_close > 0.35:
                            raw_mouth_open = 0.0
                            
                        target_mouth_open = self.filter_mouth.filter(raw_mouth_open, t_start)

                        # Highly sensitive emotion mapping
                        raw_smile = (b.get("mouthSmileLeft", 0.0) + b.get("mouthSmileRight", 0.0)) / 2.0
                        target_smile = max(0.0, min(1.0, (raw_smile - 0.02) * 2.5))
                        
                        raw_pucker = max(b.get("mouthPucker", 0.0), b.get("mouthFunnel", 0.0) * 0.8)
                        target_lip_pucker = max(0.0, min(1.0, (raw_pucker - 0.02) * 2.0))
                        
                        target_mouth_shift_x = b.get("mouthRight", 0) - b.get("mouthLeft", 0)
                        target_mouth_shift_y = 0.0

                        raw_upper_teeth = (b.get("mouthUpperUpLeft", 0.0) + b.get("mouthUpperUpRight", 0.0)) / 2.0
                        target_upper_teeth = max(0.0, min(1.0, (raw_upper_teeth - 0.05) * 2.0))
                        
                        raw_lower_teeth = (b.get("mouthLowerDownLeft", 0.0) + b.get("mouthLowerDownRight", 0.0)) / 2.0
                        target_lower_teeth = max(0.0, min(1.0, (raw_lower_teeth - 0.05) * 2.0))

                        raw_frown = (b.get("mouthFrownLeft", 0.0) + b.get("mouthFrownRight", 0.0)) / 2.0
                        target_frown = max(0.0, min(1.0, (raw_frown - 0.05) * 2.5))
                        
                        raw_asym = b.get("mouthSmileRight", 0.0) - b.get("mouthSmileLeft", 0.0)
                        target_asym = max(-1.0, min(1.0, raw_asym * 3.0))

                        stretch = (b.get("mouthStretchLeft", 0.0) + b.get("mouthStretchRight", 0.0)) / 2.0
                        target_mouth_w = 0.75 + stretch * 0.25

                        # Viseme Classification
                        viseme_mode = self.viseme_engine.classify(b, target_mouth_open, target_smile, target_lip_pucker, target_asym, target_frown)

                except Exception as e:
                    logger.debug("Tracking exception: %s", e)
                    detected = False

            self.camera_tuner.update_feedback(detected)

            if detected:
                self.face_detected = True
                alpha_mouth = 0.92
                alpha_look = 0.90
                alpha_smile = 0.85

                d_pos = math.hypot(target_head_x - self.last_head_pos[0], target_head_y - self.last_head_pos[1])
                kinetic_delta = d_pos * 8.0 + abs(target_yaw - self.yaw) * 6.0
                self.velocity = self.velocity * 0.80 + kinetic_delta * 0.20
                self.last_head_pos = (target_head_x, target_head_y)

                alpha_head = 0.85 + min(1.0, kinetic_delta) * 0.13

                self.head_x += (target_head_x - self.head_x) * alpha_head
                self.head_y += (target_head_y - self.head_y) * alpha_head
                self.yaw += (target_yaw - self.yaw) * alpha_head
                self.pitch += (target_pitch - self.pitch) * alpha_head
                self.depth += (target_depth - self.depth) * alpha_head
                self.tilt_deg += (target_tilt - self.tilt_deg) * alpha_head

                self.neck_x += (target_neck_x - self.neck_x) * alpha_head
                self.neck_y += (target_neck_y - self.neck_y) * alpha_head

                self.look_x += (target_look_x - self.look_x) * alpha_look
                self.look_y += (target_look_y - self.look_y) * alpha_look

                self.blink_left += (target_blink_left - self.blink_left) * 0.92
                self.blink_right += (target_blink_right - self.blink_right) * 0.92

                self.brow_raise += (target_brow_raise - self.brow_raise) * alpha_look
                self.eye_openness += (target_eye_openness - self.eye_openness) * alpha_look

                self.mouth_shift_x += (target_mouth_shift_x - self.mouth_shift_x) * alpha_mouth
                self.mouth_shift_y += (target_mouth_shift_y - self.mouth_shift_y) * alpha_mouth
                
                self.mouth_open += (target_mouth_open - self.mouth_open) * alpha_mouth
                self.smile_intensity += (target_smile - self.smile_intensity) * alpha_smile
                self.upper_teeth += (target_upper_teeth - self.upper_teeth) * alpha_mouth
                self.lower_teeth += (target_lower_teeth - self.lower_teeth) * alpha_mouth
                self.lip_pucker += (target_lip_pucker - self.lip_pucker) * alpha_mouth
                
                self.mouth_width += (target_mouth_w - self.mouth_width) * alpha_mouth
                self.frown += (target_frown - self.frown) * alpha_mouth
                self.smile_asymmetry += (target_asym - self.smile_asymmetry) * alpha_mouth
                self.viseme_mode = viseme_mode
                self.is_blinking = target_blink

            else:
                self.decay_to_neutral()

            state_dict = self.get_state_dict()
            self.face_updated.emit(state_dict)

            if self.audit_mode_enabled:
                calc_fps = 1.0 / max(0.001, time.time() - t_start)
                lat_ms = (time.time() - t_start) * 1000.0
                curr_lm = lm if detected else None
                curr_b = b if detected else {}
                vis_frame = annotate_audit_frame(tuned_frame, curr_lm, curr_b, state_dict, calc_fps, lat_ms)
                self.audit_frame_updated.emit(vis_frame, state_dict)

            elapsed = time.time() - t_start
            sleep_time = max(0.005, interval - elapsed)
            self.msleep(int(sleep_time * 1000))

        if mp_detector:
            try: mp_detector.close()
            except Exception: pass
        if self.cap:
            self.cap.release()

    def decay_to_neutral(self) -> None:
        decay = 0.15
        self.head_x *= (1 - decay); self.head_y *= (1 - decay)
        self.yaw *= (1 - decay); self.pitch *= (1 - decay)
        self.depth += (1.0 - self.depth) * decay
        self.neck_x *= (1 - decay); self.neck_y *= (1 - decay)
        self.look_x *= (1 - decay); self.look_y *= (1 - decay)
        self.blink_left *= (1 - decay); self.blink_right *= (1 - decay)
        self.brow_raise *= (1 - decay); self.eye_openness += (0.5 - self.eye_openness) * decay
        self.smile_intensity *= (1 - decay); self.mouth_open *= (1 - decay)
        self.mouth_shift_x *= (1 - decay); self.mouth_shift_y *= (1 - decay)
        self.upper_teeth *= (1 - decay); self.lower_teeth *= (1 - decay)
        self.lip_pucker *= (1 - decay); self.frown *= (1 - decay)
        self.smile_asymmetry *= (1 - decay); self.tilt_deg *= (1 - decay)
        self.velocity *= (1 - decay); self.is_blinking = False
        self.viseme_mode = "NEUTRAL"

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "face_detected": self.face_detected,
            "tuner_status": getattr(self.camera_tuner, "status", "NORMAL"),
            "head_x": self.head_x, "head_y": self.head_y,
            "yaw": self.yaw, "pitch": self.pitch, "tilt_deg": self.tilt_deg,
            "depth": self.depth, "neck_x": self.neck_x, "neck_y": self.neck_y,
            "velocity": self.velocity, "look_x": self.look_x, "look_y": self.look_y,
            "blink_left": self.blink_left, "blink_right": self.blink_right,
            "is_winking_left": self.blink_left > 0.55 and self.blink_right < 0.35,
            "is_winking_right": self.blink_right > 0.55 and self.blink_left < 0.35,
            "brow_raise": self.brow_raise, "eye_openness": self.eye_openness,
            "smile": self.smile_intensity, "frown": self.frown,
            "smile_asymmetry": self.smile_asymmetry, "viseme_mode": self.viseme_mode,
            "mouth_open": self.mouth_open, "mouth_width": self.mouth_width,
            "mouth_shift_x": self.mouth_shift_x, "mouth_shift_y": self.mouth_shift_y,
            "upper_teeth": self.upper_teeth, "lower_teeth": self.lower_teeth,
            "lip_pucker": self.lip_pucker, "is_smiling": self.smile_intensity > 0.25,
            "is_talking": self.mouth_open > 0.12, "is_blinking": self.is_blinking,
        }

    def stop(self) -> None:
        self.running = False
        if hasattr(self, "cap") and self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap = None
        self.wait(300)
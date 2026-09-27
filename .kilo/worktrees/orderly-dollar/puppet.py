#!/usr/bin/env python3
import sys
import os
import math
import time
import ctypes
import random

from PySide6.QtWidgets import QApplication, QWidget, QMenu
from PySide6.QtCore import Qt, QPoint, QPointF, QRectF, QTimer, QEvent, QFileSystemWatcher, QSettings
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPainterPath, QAction,
    QRadialGradient, QPixmap, QImage, QIcon, QPolygonF, QCursor, QTransform
)

from face_tracker import FaceTrackerThread
from cloath_physics import VerletClothCape, draw_pixelated_cape, BLOSSOM_X6, CAPE_PALETTES
from assets.headwears import load_all_headwears

def make_permanent_always_on_top(widget):
    if sys.platform == "darwin":
        try:
            from ctypes import c_void_p, c_char_p, c_int, c_ulong, c_bool
            objc = ctypes.cdll.LoadLibrary('/usr/lib/libobjc.A.dylib')
            objc.sel_registerName.restype = c_void_p
            objc.sel_registerName.argtypes = [c_char_p]
            view_ptr = c_void_p(int(widget.winId()))
            sel_window = objc.sel_registerName(b'window')
            msg_send_ptr = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(c_void_p, c_void_p, c_void_p))
            window_ptr = msg_send_ptr(view_ptr, sel_window)
            if window_ptr:
                sel_setLevel = objc.sel_registerName(b'setLevel:')
                msg_setLevel = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_int))
                msg_setLevel(window_ptr, sel_setLevel, 25)
                sel_setBehavior = objc.sel_registerName(b'setCollectionBehavior:')
                msg_setBehavior = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_ulong))
                msg_setBehavior(window_ptr, sel_setBehavior, 1 | 16 | 64 | 256)
                sel_setHides = objc.sel_registerName(b'setHidesOnDeactivate:')
                msg_setHides = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(None, c_void_p, c_void_p, c_bool))
                msg_setHides(window_ptr, sel_setHides, False)
                return True
        except Exception:
            pass
    return False

class MoodKnowledgeBase:
    ONTOLOGY = {
        "NEUTRAL_CALM": {"gloss_boost": 1.0, "pupil_scale": 1.0, "glow_alpha": 40, "fluid_elasticity": 1.0},
        "HAPPY_WARM": {"gloss_boost": 1.25, "pupil_scale": 1.08, "glow_alpha": 65, "fluid_elasticity": 1.15},
        "JOYFUL_BEAMING": {"gloss_boost": 1.45, "pupil_scale": 1.18, "glow_alpha": 85, "fluid_elasticity": 1.35},
        "TALKING_CHATTY": {"gloss_boost": 1.15, "pupil_scale": 1.05, "glow_alpha": 50, "fluid_elasticity": 1.1},
        "SURPRISED_AWE": {"gloss_boost": 1.6, "pupil_scale": 1.35, "glow_alpha": 90, "fluid_elasticity": 1.4},
        "CURIOUS_INQUISITIVE": {"gloss_boost": 1.3, "pupil_scale": 1.15, "glow_alpha": 60, "fluid_elasticity": 1.2},
        "SKEPTICAL_DOUBT": {"gloss_boost": 0.9, "pupil_scale": 0.85, "glow_alpha": 45, "fluid_elasticity": 0.9},
        "FOCUSED_INTENSE": {"gloss_boost": 1.1, "pupil_scale": 0.9, "glow_alpha": 55, "fluid_elasticity": 0.95},
        "PLAYFUL_SMUG": {"gloss_boost": 1.35, "pupil_scale": 1.1, "glow_alpha": 70, "fluid_elasticity": 1.25},
        "THOUGHTFUL_CONTEMPLATIVE": {"gloss_boost": 1.05, "pupil_scale": 0.95, "glow_alpha": 45, "fluid_elasticity": 1.0},
        "SLEEPY_DROWSY": {"gloss_boost": 0.7, "pupil_scale": 0.8, "glow_alpha": 30, "fluid_elasticity": 0.8},
        "ECSTATIC_ACROBATIC": {"gloss_boost": 1.8, "pupil_scale": 1.3, "glow_alpha": 100, "fluid_elasticity": 1.6}
    }

    def __init__(self):
        self.active_mood = "NEUTRAL_CALM"
        self.mood_confidence = 1.0
        self.last_switch_time = time.time()
        self.smooth_gloss = 1.0
        self.smooth_pupil = 1.0
        self.smooth_glow = 40.0
        self.smooth_elasticity = 1.0
        self.mood_weights = {k: (1.0 if k == "NEUTRAL_CALM" else 0.0) for k in self.ONTOLOGY}

    def evaluate(self, state):
        smile = state.get("smile", 0.0)
        mouth_open = state.get("mouth_open", 0.0)
        brow_raise = state.get("brow_raise", 0.0)
        tilt = abs(state.get("tilt_deg", 0.0))
        pitch = state.get("pitch", 0.0)
        look_x = state.get("look_x", 0.0)
        look_y = state.get("look_y", 0.0)
        is_blinking = state.get("is_blinking", False)
        velocity = state.get("velocity", 0.0)
        is_spinning = state.get("is_spinning", False)
        frown = state.get("frown", 0.0)
        smile_asym = state.get("smile_asymmetry", 0.0)
        lip_pucker = state.get("lip_pucker", 0.0)
        eye_openness = state.get("eye_openness", 0.5)
        is_winking_l = state.get("is_winking_left", False)
        is_winking_r = state.get("is_winking_right", False)
        viseme = state.get("viseme_mode", "NEUTRAL")
        upper_teeth = state.get("upper_teeth", 0.0)

        scores = {}
        if is_spinning or (velocity > 1.2 and smile > 0.45):
            scores["ECSTATIC_ACROBATIC"] = 0.95 + min(0.05, velocity * 0.1)
        elif velocity > 0.8 and smile > 0.35:
            scores["ECSTATIC_ACROBATIC"] = 0.65 + velocity * 0.2 + smile * 0.15

        brow_hi = max(0.0, brow_raise)
        eye_wide = max(0.0, (eye_openness - 0.60) * 2.0)
        jaw_drop = max(0.0, mouth_open)
        vis_surprise = 0.25 if viseme in ["OPEN_JAW", "O_PHONEME", "ROUNDED_O", "OPEN_VOWEL"] else 0.0
        if (brow_hi > 0.20 or eye_wide > 0.20) and jaw_drop > 0.18:
            scores["SURPRISED_AWE"] = 0.45 + brow_hi * 0.30 + eye_wide * 0.20 + jaw_drop * 0.30 + vis_surprise

        grin_boost = 0.25 if viseme in ["WIDE_GRIN", "TRIANGLE_SMILE", "SPREAD_E_I"] else 0.0
        if smile > 0.50 or (smile > 0.35 and upper_teeth > 0.20):
            scores["JOYFUL_BEAMING"] = 0.45 + smile * 0.38 + upper_teeth * 0.18 + grin_boost

        if smile > 0.18 and mouth_open < 0.22:
            scores["HAPPY_WARM"] = 0.45 + smile * 0.45 - mouth_open * 0.25

        talk_boost = 0.28 if viseme in ["TALKING", "OPEN_JAW", "O_PHONEME", "OPEN_VOWEL", "POSTALVEOLAR", "LABIODENTAL", "ALVEOLAR", "DENTAL", "SPREAD_E_I"] else 0.0
        if mouth_open > 0.12 and smile < 0.45:
            scores["TALKING_CHATTY"] = 0.45 + mouth_open * 0.38 + talk_boost

        smirk_boost = 0.38 if viseme == "SMIRK" else 0.0
        asym_val = abs(smile_asym)
        wink_boost = 0.30 if (is_winking_l or is_winking_r) else 0.0
        pucker_boost = lip_pucker * 0.25
        if asym_val > 0.18 or smirk_boost > 0.0 or wink_boost > 0.0 or (0.15 < smile < 0.50 and tilt > 12.0):
            scores["PLAYFUL_SMUG"] = 0.45 + smirk_boost + asym_val * 0.40 + wink_boost + pucker_boost + (tilt / 45.0) * 0.20

        if (brow_hi > 0.18 and tilt > 9.0) or (tilt > 14.0 and abs(look_x) > 0.25):
            scores["CURIOUS_INQUISITIVE"] = 0.45 + (brow_hi * 0.25) + (tilt / 45.0) * 0.25 + abs(look_x) * 0.20

        frown_boost = (0.35 if viseme == "FROWN" else 0.0) + frown * 0.45
        squint_boost = max(0.0, (0.50 - eye_openness) * 0.70)
        furrow_boost = max(0.0, -brow_raise) * 0.40
        if frown_boost > 0.20 or (furrow_boost > 0.18 and tilt > 8.0) or (squint_boost > 0.20 and tilt > 10.0):
            scores["SKEPTICAL_DOUBT"] = 0.45 + frown_boost + squint_boost + furrow_boost + (tilt / 45.0) * 0.15

        firm_lips = max(0.0, 1.0 - mouth_open * 4.0) * 0.15
        steady_gaze = max(0.0, 1.0 - math.hypot(look_x, look_y) * 2.0) * 0.20
        if furrow_boost > 0.15 and steady_gaze > 0.10 and mouth_open < 0.12 and tilt < 10.0:
            scores["FOCUSED_INTENSE"] = 0.45 + furrow_boost + squint_boost + steady_gaze + firm_lips

        up_gaze = max(0.0, -look_y)
        side_gaze = abs(look_x)
        pensive_gaze = up_gaze * 0.60 + side_gaze * 0.25
        if pensive_gaze > 0.22 and mouth_open < 0.14 and smile < 0.20:
            scores["THOUGHTFUL_CONTEMPLATIVE"] = 0.45 + pensive_gaze * 0.40 + (tilt / 45.0) * 0.15

        droop = max(0.0, (0.42 - eye_openness) * 1.3)
        down_pitch = max(0.0, pitch * 0.40)
        if (droop > 0.20 or is_blinking) and down_pitch > 0.15 and mouth_open < 0.12:
            scores["SLEEPY_DROWSY"] = 0.45 + droop + down_pitch + (0.20 if is_blinking else 0.0)

        scores["NEUTRAL_CALM"] = 0.42

        temperature = 0.22
        max_s = max(scores.values())
        exp_scores = {k: math.exp((s - max_s) / temperature) for k, s in scores.items()}
        sum_exp = sum(exp_scores.values())
        for k in self.ONTOLOGY:
            w_k = exp_scores.get(k, 0.0) / sum_exp
            self.mood_weights[k] = self.mood_weights.get(k, 0.0) * 0.85 + w_k * 0.15

        target_gloss = sum(self.mood_weights[k] * self.ONTOLOGY[k]["gloss_boost"] for k in self.ONTOLOGY)
        target_pupil = sum(self.mood_weights[k] * self.ONTOLOGY[k]["pupil_scale"] for k in self.ONTOLOGY)
        target_glow = sum(self.mood_weights[k] * self.ONTOLOGY[k]["glow_alpha"] for k in self.ONTOLOGY)
        target_elasticity = sum(self.mood_weights[k] * self.ONTOLOGY[k]["fluid_elasticity"] for k in self.ONTOLOGY)

        alpha = 0.12
        self.smooth_gloss += (target_gloss - self.smooth_gloss) * alpha
        self.smooth_pupil += (target_pupil - self.smooth_pupil) * alpha
        self.smooth_glow += (target_glow - self.smooth_glow) * alpha
        self.smooth_elasticity += (target_elasticity - self.smooth_elasticity) * alpha

        best_mood = max(scores, key=scores.get)
        best_score = scores[best_mood]
        now = time.time()
        if best_mood != self.active_mood:
            if best_score > (self.mood_confidence + 0.12) or (now - self.last_switch_time > 0.40):
                self.active_mood = best_mood
                self.mood_confidence = best_score
                self.last_switch_time = now
        else:
            self.mood_confidence = best_score

        active_info = self.ONTOLOGY[self.active_mood].copy()
        active_info["gloss_boost"] = self.smooth_gloss
        active_info["pupil_scale"] = self.smooth_pupil
        active_info["glow_alpha"] = self.smooth_glow
        active_info["fluid_elasticity"] = self.smooth_elasticity

        return {"mood_key": self.active_mood, "info": active_info, "confidence": self.mood_confidence, "weights": self.mood_weights}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_IMG_PATH = os.path.join(BASE_DIR, "base.png")

def get_software_icon():
    icon_paths = [os.path.join(BASE_DIR, "icon.png"), os.path.join(BASE_DIR, "cap.png")]
    for p in icon_paths:
        if os.path.exists(p):
            pix = QPixmap(p)
            if not pix.isNull():
                return QIcon(pix)
    return QIcon()

BASE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAYAAADDPmHLAAAD9UlEQVR4nO2dy5HcMBBDMU7ACfjm"
    "/EPyzQk4gvGpt2Z29CElkv0B3skXuagCGtBKHOkBYp7P59P+/Xg8Hp5r8YLypIF38Q1GE/zwXoDw"
    "hc7xwPb0G2wpQHWywLH4BpMJaE4UaBPfYDEB/TXA75+/vJfgCoXLge3pfxX/z7+/H8cwpED5EwQ+"
    "xT+a+u9GqG4Cugo4i3y2SijtbuB9+nvEfU2CyilQ9sSA6+IbDCagqwDxTklXA/en36ieAuVOCBgn"
    "vlHZBKVOBhgvvlHVBLoGIKeMk4F5029UTIESJwHMF9+oZgJVADnpHQysm36jUgqkXjywXnyjiglU"
    "AeSkdS7gN/1GhRRIuWjAX3wjuwlUAeSkcywQZ/qNzCmQarFAPPGNrCZQBZCTxqlA3Ok3MqZAikUC"
    "8cU3splAFUBOeIcCeabfyJQCoRcH5BPfyGICVQA5YZ0J5J1+I0MKhFwUkF98I7oJVAHkhHMkUGf6"
    "jcgpEGoxQD3xjagmUAWQE8aJQN3pNyKmQIhFAPXFN6KZQBVAjrsDAZ7pNyKlgBLAgVeT97y7cAbu"
    "BmCb/mi4xg+7+BGqwD0BmIlQBW4GYJ/+KLjEjsR/x7MKVAEB8KyC5QbQ9MdCCRAErxRY2jea/nNW"
    "Xw8sSwCJHxNVQDBWV8ESA2j646IECMjKFJhuAE1/bKZeZUr8e6z4i0AVEJgVVTDNAJr+HEyJFYk/"
    "lplVoAogZ3gCaPrnMCsFlADkDE0ATf9cZqTAMANI/DWMNoEqgJwhCaDpX8vIFFACkHM7ATT9PoxK"
    "ASUAObcSQNPvy4gUUAKQczkBNP0xuJsClxJA4tdBFZCcu5tGuiND0x+Tq1XQlQASvx6qgCJcrYJm"
    "A2j6a6IEKMSVFGgygKa/LkqAYvSmwKkBNP21UQIUpCcFDg2g6a+PEoAcGaAorTWwawDFPwdKAHJk"
    "AHJkgMK0XAdsGkD9z4MSgBwZgBwZgBwZgIitC8EPA3h/xkysRQlAjgxQmNet4nucGqDlPxF5UQKQ"
    "IwMQsfWLoQ8DeH/NWoyhtbqbEkDXAXVRBZAjAxRkK7H3qn2377fuCOrRcHx6xAcOEmDrIF0L1EMV"
    "UIje6Qca3hCiKsjBFfGBxlfEyASxuSo+0PGOIJkgJnfEBzpfEiUTxOGu8F/H9B4gE/hx9FfY1Vv4"
    "lw7a2zUkI8xhhvBfx985+Gz7mAxxnbN7LqG+GdSyj1BmOKblJtuMJ7XTvhp2BrMheu+oznxEv+zr"
    "4T1UMsfV2+er9mUs2/wxart5VHPcfU7itRHHbffP6N8frDDGqIdhkXZdhVnIKzN+nNJjkJFPPSOJ"
    "vUXoxW0R9ZdL0YXeI+Wit1hhjKwiH/EfDGwoGRxDKecAAAAASUVORK5CYII="
)

BASE_CONTOUR_NORM = [
    (0.0083, -0.5), (-0.0083, -0.5), (-0.25, -0.1719), (-0.475, 0.1875),
    (-0.4917, 0.2422), (-0.4917, 0.2734), (-0.5, 0.2812), (-0.4917, 0.3516),
    (-0.425, 0.4219), (-0.375, 0.4453), (-0.35, 0.4453), (-0.2833, 0.4688),
    (-0.175, 0.4766), (-0.1167, 0.4922), (0.0833, 0.4922), (0.0917, 0.4844),
    (0.1917, 0.4766), (0.2, 0.4688), (0.2333, 0.4688), (0.2417, 0.4609),
    (0.275, 0.4609), (0.3333, 0.4453), (0.4333, 0.3984), (0.4833, 0.3438),
    (0.4833, 0.3203), (0.4917, 0.3125), (0.4833, 0.2344), (0.4667, 0.1797),
    (0.4333, 0.1172), (0.4167, 0.1016), (0.3167, -0.0781),
]

BLOSSOM_X6 = {
    "BLACK": "#000000",
    "DEEP_BERRY": "#b3003c",
    "RUBY": "#cd1351",
    "CRIMSON": "#e72e4a",
    "CORAL": "#ff4b4b",
    "WHITE": "#ffffff",
}

THEMES = {
    "base": {"name": "🖤 Blossom Void (Base)", "base_color": "#000000", "deep_color": "#000000", "rim_color": "#ffffff", "highlight_color": "#ff4b4b", "eye_color": "#ffffff", "mouth_color": "#ffffff"},
    "blossom_crimson": {"name": "🌺 Blossom Crimson (3D)", "base_color": "#e72e4a", "deep_color": "#b3003c", "rim_color": "#ffffff", "highlight_color": "#ff4b4b", "eye_color": "#ffffff", "mouth_color": "#ffffff"},
    "blossom_ruby": {"name": "💎 Blossom Ruby (3D)", "base_color": "#cd1351", "deep_color": "#000000", "rim_color": "#ffffff", "highlight_color": "#ff4b4b", "eye_color": "#ffffff", "mouth_color": "#ffffff"},
    "blossom_berry": {"name": "🍇 Blossom Berry (3D)", "base_color": "#b3003c", "deep_color": "#000000", "rim_color": "#ffffff", "highlight_color": "#ff4b4b", "eye_color": "#ffffff", "mouth_color": "#ffffff"}
}

class FloatingCapInstance:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.rot = 0.0
        self.v_rot = 0.0
        self.hover_phase = 0.0
        self.initialized = False

    def update(self, dt, target_x, target_y, target_rot, drag_vx, drag_vy, scale, is_dragging):
        self.hover_phase += dt
        eff_target_x = target_x
        eff_target_y = target_y + math.sin(self.hover_phase * 3.2) * (1.8 * scale)

        if not self.initialized:
            self.x = eff_target_x
            self.y = eff_target_y
            self.rot = target_rot
            self.initialized = True
            return

        dx = eff_target_x - self.x
        dy = eff_target_y - self.y
        dist = math.hypot(dx, dy)

        if is_dragging:
            k_spring = 12.0
            damping = 0.90
        else:
            k_spring = 28.0 if dist < 25.0 * scale else 44.0
            damping = 0.82

        ax = dx * k_spring
        ay = dy * k_spring

        if is_dragging:
            ax -= drag_vx * 60.0
            ay -= drag_vy * 60.0

        self.vx = (self.vx + ax * dt) * damping
        self.vy = (self.vy + ay * dt) * damping
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        min_cap_y = 6.0 * scale
        if self.y < min_cap_y:
            self.y = min_cap_y
            self.vy = max(0.0, self.vy)

        wind_torque = -self.vx * 0.15
        target_angle = target_rot + wind_torque + math.sin(self.hover_phase * 2.5) * 1.5
        rot_error = target_angle - self.rot
        self.v_rot = (self.v_rot + rot_error * 32.0 * dt) * 0.80
        self.rot += self.v_rot * dt


class FacePuppetPet(QWidget):
    def __init__(self, theme_key="base", start_tracker=True):
        super().__init__()
        
        self.settings = QSettings("Inklet", "FacePuppet")
        saved_theme = self.settings.value("theme_key", theme_key, type=str)
        
        self.theme_key = saved_theme if saved_theme in THEMES else "base"
        self.theme = THEMES[self.theme_key]

        if os.path.exists(BASE_IMG_PATH):
            self.base_pixmap = QPixmap(BASE_IMG_PATH)
            self.base_raw_pixmap = QPixmap(BASE_IMG_PATH)
        else:
            import base64
            img_bytes = base64.b64decode(BASE_PNG_B64)
            pix = QPixmap()
            pix.loadFromData(img_bytes)
            self.base_pixmap = pix
            self.base_raw_pixmap = pix

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)

        self.app_icon = get_software_icon()
        if not self.app_icon.isNull():
            self.setWindowIcon(self.app_icon)

        self.resize(300, 300)
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else QRectF(0, 0, 1200, 800)
        self.move(int(avail.x() + avail.width() - 330), int(avail.y() + avail.height() - 330))

        self.watcher = QFileSystemWatcher([os.path.abspath(__file__)], self)
        self.watcher.fileChanged.connect(self.hot_reload)

        self.target_data = {
            "face_detected": False, "head_x": 0.0, "head_y": 0.0, "yaw": 0.0, "pitch": 0.0, "tilt_deg": 0.0, "depth": 1.0,
            "neck_x": 0.0, "neck_y": 0.0, "velocity": 0.0, "look_x": 0.0, "look_y": 0.0, "blink_left": 0.0, "blink_right": 0.0,
            "brow_raise": 0.0, "eye_openness": 0.5, "smile": 0.0, "mouth_open": 0.0, "mouth_width": 0.75, "mouth_shift_x": 0.0,
            "mouth_shift_y": 0.0, "upper_teeth": 0.0, "lower_teeth": 0.0, "lip_pucker": 0.0, "frown": 0.0, "smile_asymmetry": 0.0,
            "viseme_mode": "NEUTRAL", "is_blinking": False,
        }

        self.head_x = 0.0
        self.head_y = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.tilt_deg = 0.0
        self.depth = 1.0
        self.neck_x = 0.0
        self.neck_y = 0.0
        self.look_x = 0.0
        self.look_y = 0.0
        self.blink_left = 0.0
        self.blink_right = 0.0
        self.brow_raise = 0.0
        self.eye_openness = 0.5
        self.smile = 0.0
        self.mouth_open = 0.0
        self.mouth_width = 0.75
        self.mouth_shift_x = 0.0
        self.mouth_shift_y = 0.0
        self.upper_teeth = 0.0
        self.lower_teeth = 0.0
        self.lip_pucker = 0.0
        self.frown = 0.0
        self.smile_asymmetry = 0.0
        self.viseme_mode = "NEUTRAL"
        self.blink_val = 0.0
        self.face_detected = False

        self.manual_blink_l_timer = 0.0
        self.manual_blink_r_timer = 0.0
        self.last_eye_l = QPointF(0, 0)
        self.last_eye_r = QPointF(0, 0)
        self.eye_hit_radius = 25.0

        self.rot_x = 0.0
        self.is_flipping_x = False
        self.flip_start_time = 0.0
        self.flip_duration = 0.62

        self.squash_x = 1.0
        self.squash_y = 1.0
        self.vel_x = 0.0
        self.vel_y = 0.0

        self.mood_engine = MoodKnowledgeBase()
        self.current_mood = self.mood_engine.evaluate({})

        self.dragging = False
        self.resizing = False
        self.rotating = False
        self.manual_rotation_z = 0.0
        self.rotate_start_angle = 0.0
        self.rotate_initial_angle = 0.0
        self.right_press_pos = None
        self.drag_start_global = QPoint()
        self.window_start_pos = QPoint()
        self.resize_start_pos = QPoint()
        self.resize_start_size = None
        self.resize_corner_margin = 28
        self.click_through_mode = False

        self.drag_anim_val = 0.0
        self.drag_phase = 0.0
        self.drag_vel_x = 0.0
        self.drag_vel_y = 0.0

        self.show_pointers = False
        self.hover_top_handle = False
        self.hover_right_handle = False
        self.scaling_active = False
        self.handle_rotating = False
        self.scale_start_mouse_y = 0.0
        self.scale_start_dim = 300
        self.last_head_cx = 150.0
        self.last_head_cy = 150.0
        self.last_head_rx = 60.0
        self.last_head_ry = 60.0
        self.last_head_rot = 0.0

        self.cape_sim = VerletClothCape(cols=3, rows=10)
        self.last_global_pos = self.pos()
        self.last_collar_gx = 0.0
        self.last_collar_gy = 0.0
        self.collar_initialized = False
        self.cape_vel_x = 0.0
        self.cape_vel_y = 0.0
        self.cape_angle = 0.0
        self.cape_avel = 0.0
        self.cape_curl = 0.0
        self.cape_stretch = 1.0
        self.cape_anim_state = "IDLE"
        self.cape_anim_timer = 0.0
        self.next_cape_anim_time = time.time() + 3.0

        self._cg_event_btn = None
        if sys.platform == "darwin":
            try:
                cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
                cg.CGEventSourceButtonState.restype = ctypes.c_bool
                cg.CGEventSourceButtonState.argtypes = [ctypes.c_int32, ctypes.c_uint32]
                self._cg_event_btn = cg.CGEventSourceButtonState
            except Exception:
                pass

        self.cap_instance = FloatingCapInstance()
        self.hat_catalogue = []
        self.hat_flipped = False

        self.cape_palettes = CAPE_PALETTES

        self.show_inventory = False
        self.hover_inventory_slot = None
        self.hover_inventory_idx = None
        self.hover_pencil_handle = False
        self.inventory_rect = QRectF()
        self.inventory_slot_rects = []
        self.inventory_hat_slot_rects = []
        self.inventory_cape_slot_rects = []
        self.inventory_shadow_slot_rects = []
        
        self.particles = []

        self.show_cap = self.settings.value("show_cap", True, type=bool)
        self.show_cape = self.settings.value("show_cape", True, type=bool)
        self.show_shadow = self.settings.value("show_shadow", True, type=bool)
        
        self.active_hat_idx = self.settings.value("active_hat_idx", 0, type=int)
        self.active_cape_color = self.settings.value("active_cape_color", "red", type=str)
        self.active_shadow_color = self.settings.value("active_shadow_color", "red", type=str)

        self.init_hat_catalogue()

        if start_tracker:
            self.tracker = FaceTrackerThread(target_fps=30)
            self.tracker.face_updated.connect(self.on_face_updated)
            self.tracker.start()
        else:
            self.tracker = None

        self.timer_72fps = QTimer(self)
        self.timer_72fps.setInterval(int(1000.0 / 72.0))
        self.timer_72fps.timeout.connect(self.physics_step_72fps)
        self.timer_72fps.start()

    def save_settings(self):
        self.settings.setValue("show_cap", self.show_cap)
        self.settings.setValue("show_cape", self.show_cape)
        self.settings.setValue("show_shadow", self.show_shadow)
        self.settings.setValue("active_hat_idx", self.active_hat_idx)
        self.settings.setValue("active_cape_color", self.active_cape_color)
        self.settings.setValue("active_shadow_color", self.active_shadow_color)
        self.settings.setValue("theme_key", self.theme_key)

    def hot_reload(self):
        if hasattr(self, "tracker") and self.tracker:
            self.tracker.stop()
        os.execv(sys.executable, ['python3'] + sys.argv)

    def init_hat_catalogue(self):
        self.hat_catalogue = load_all_headwears()
        self.set_active_hat(self.active_hat_idx)

    def set_active_hat(self, idx):
        if not self.hat_catalogue:
            self.cap_pixmap = None
            self.cap_silhouette = None
            return
        self.active_hat_idx = max(0, min(len(self.hat_catalogue) - 1, idx))
        hat = self.hat_catalogue[self.active_hat_idx]
        self.cap_pixmap = hat["pixmap"]
        self.cap_silhouette = hat["white_sil"]

    def get_current_hat_tuck_ny(self):
        if not self.show_cap or not hasattr(self, "hat_catalogue") or not self.hat_catalogue:
            return -0.50
        hat = self.hat_catalogue[self.active_hat_idx]
        return hat.get("tuck_y", -0.28)

    def on_face_updated(self, data):
        self.target_data = data

    def trigger_360_flip_x(self):
        if not self.is_flipping_x:
            self.is_flipping_x = True
            self.flip_start_time = time.time()

    def get_colored_hat_shadow(self):
        color_key = getattr(self, "active_shadow_color", "red")
        if hasattr(self, "_cached_hat_shadow_key") and self._cached_hat_shadow_key == color_key and hasattr(self, "_cached_hat_shadow"):
            return self._cached_hat_shadow
        shadow_pal = CAPE_PALETTES.get(color_key, CAPE_PALETTES["red"])
        shadow_color = QColor(shadow_pal["dark"])
        colored_shadow = QPixmap(self.cap_silhouette.size())
        colored_shadow.fill(Qt.GlobalColor.transparent)
        p = QPainter(colored_shadow)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawPixmap(0, 0, self.cap_silhouette)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        p.fillRect(colored_shadow.rect(), shadow_color)
        p.end()
        self._cached_hat_shadow_key = color_key
        self._cached_hat_shadow = colored_shadow
        return colored_shadow

    def physics_step_72fps(self):
        # 1. Z-Depth Scaling: Enlarge when leaning in, shrink when leaning out
        dynamic_depth_scale = max(0.4, min(1.8, 0.5 + self.depth * 0.5))
        scale = (min(self.width(), self.height()) / 280.0) * dynamic_depth_scale
        
        dt = 1.0 / 72.0
        td = self.target_data
        self.face_detected = td.get("face_detected", False)

        # Decay manual blink timers for click-to-blink
        self.manual_blink_l_timer = max(0.0, self.manual_blink_l_timer - dt * 3.5)
        self.manual_blink_r_timer = max(0.0, self.manual_blink_r_timer - dt * 3.5)

        if self.is_flipping_x:
            elapsed = time.time() - self.flip_start_time
            progress = min(1.0, elapsed / self.flip_duration)
            t = progress
            ease = 3.0 * t * t - 2.0 * t * t * t
            self.rot_x = ease * 360.0
            if progress >= 1.0:
                self.rot_x = 0.0
                self.is_flipping_x = False
        else:
            self.rot_x = 0.0

        target_yaw = td.get("yaw", 0.0)
        target_pitch = td.get("pitch", 0.0)
        target_tilt = td.get("tilt_deg", 0.0)
        target_head_x = td.get("head_x", 0.0)
        target_head_y = td.get("head_y", 0.0)

        k_pose = 0.28
        k_smooth = 0.26
        k_fast = 0.38

        self.head_x += (target_head_x - self.head_x) * k_smooth
        self.head_y += (target_head_y - self.head_y) * k_smooth
        self.yaw += (target_yaw - self.yaw) * k_pose
        self.pitch += (target_pitch - self.pitch) * k_pose
        self.tilt_deg += (target_tilt - self.tilt_deg) * k_smooth
        self.depth += (td.get("depth", 1.0) - self.depth) * k_smooth

        self.neck_x += (td.get("neck_x", 0.0) - self.neck_x) * k_smooth
        self.neck_y += (td.get("neck_y", 0.0) - self.neck_y) * k_smooth

        if target_head_x == 0.0 and abs(self.head_x) < 0.015: self.head_x = 0.0
        if target_head_y == 0.0 and abs(self.head_y) < 0.015: self.head_y = 0.0
        if target_yaw == 0.0 and abs(self.yaw) < 0.015: self.yaw = 0.0
        if target_pitch == 0.0 and abs(self.pitch) < 0.015: self.pitch = 0.0
        if target_tilt == 0.0 and abs(self.tilt_deg) < 0.3: self.tilt_deg = 0.0

        # Hat flipped logic inverted
        if self.yaw < -0.05:
            self.hat_flipped = True
        elif self.yaw > 0.05:
            self.hat_flipped = False

        self.look_x += (td.get("look_x", 0.0) - self.look_x) * min(0.95, k_fast + 0.20)
        self.look_y += (td.get("look_y", 0.0) - self.look_y) * min(0.95, k_fast + 0.20)

        target_blink_l = td.get("blink_left", 0.0)
        target_blink_r = td.get("blink_right", 0.0)
        k_blink_l = 0.92 if target_blink_l > self.blink_left else 0.60
        k_blink_r = 0.92 if target_blink_r > self.blink_right else 0.60
        self.blink_left += (target_blink_l - self.blink_left) * k_blink_l
        self.blink_right += (target_blink_r - self.blink_right) * k_blink_r

        self.brow_raise += (td.get("brow_raise", 0.0) - self.brow_raise) * k_fast
        self.eye_openness += (td.get("eye_openness", 0.5) - self.eye_openness) * k_fast
        self.smile += (td.get("smile", 0.0) - self.smile) * k_fast
        self.mouth_open += (td.get("mouth_open", 0.0) - self.mouth_open) * min(0.95, k_fast + 0.25)
        self.mouth_width += (td.get("mouth_width", 0.75) - self.mouth_width) * k_fast
        self.mouth_shift_x += (td.get("mouth_shift_x", 0.0) - self.mouth_shift_x) * k_fast
        self.mouth_shift_y += (td.get("mouth_shift_y", 0.0) - self.mouth_shift_y) * k_fast
        self.upper_teeth += (td.get("upper_teeth", 0.0) - self.upper_teeth) * k_fast
        self.lower_teeth += (td.get("lower_teeth", 0.0) - self.lower_teeth) * k_fast
        self.lip_pucker += (td.get("lip_pucker", 0.0) - self.lip_pucker) * k_fast
        self.frown += (td.get("frown", 0.0) - self.frown) * k_fast
        self.smile_asymmetry += (td.get("smile_asymmetry", 0.0) - self.smile_asymmetry) * k_fast
        self.viseme_mode = td.get("viseme_mode", "NEUTRAL")

        if td.get("mouth_open", 0.0) == 0.0:
            if self.mouth_open < 0.04: self.mouth_open = 0.0
            self.lip_pucker = 0.0
        if td.get("smile", 0.0) == 0.0 and self.smile < 0.02: self.smile = 0.0
        if td.get("frown", 0.0) == 0.0 and self.frown < 0.02: self.frown = 0.0
        if td.get("smile_asymmetry", 0.0) == 0.0 and abs(self.smile_asymmetry) < 0.02: self.smile_asymmetry = 0.0
        if td.get("upper_teeth", 0.0) == 0.0 and self.upper_teeth < 0.02: self.upper_teeth = 0.0
        if td.get("lower_teeth", 0.0) == 0.0 and self.lower_teeth < 0.02: self.lower_teeth = 0.0
        if td.get("lip_pucker", 0.0) == 0.0: self.lip_pucker = 0.0
        if td.get("blink_left", 0.0) == 0.0 and self.blink_left < 0.04: self.blink_left = 0.0
        if td.get("blink_right", 0.0) == 0.0 and self.blink_right < 0.04: self.blink_right = 0.0
        if td.get("look_x", 0.0) == 0.0 and abs(self.look_x) < 0.03: self.look_x = 0.0
        if td.get("look_y", 0.0) == 0.0 and abs(self.look_y) < 0.03: self.look_y = 0.0
        if td.get("brow_raise", 0.0) == 0.0 and abs(self.brow_raise) < 0.03:
            self.brow_raise = 0.0
            self.eye_openness = 0.5
        if td.get("mouth_shift_x", 0.0) == 0.0 and abs(self.mouth_shift_x) < 0.01: self.mouth_shift_x = 0.0
        if td.get("mouth_shift_y", 0.0) == 0.0 and abs(self.mouth_shift_y) < 0.01: self.mouth_shift_y = 0.0

        if td.get("is_blinking", False):
            self.blink_val = min(1.0, self.blink_val + 0.32)
        else:
            self.blink_val = max(0.0, self.blink_val - 0.22)

        d_hx = td.get("head_x", 0.0) - self.head_x
        d_hy = td.get("head_y", 0.0) - self.head_y
        vel_mag = math.hypot(d_hx, d_hy)
        if vel_mag < 0.045:
            target_squash_y = 1.0
            target_squash_x = 1.0
        else:
            target_squash_y = max(0.88, min(1.15, 1.0 - d_hy * 0.35 + abs(d_hx) * 0.12))
            target_squash_x = max(0.88, min(1.15, 1.0 / target_squash_y))
        self.squash_x += (target_squash_x - self.squash_x) * 0.15
        self.squash_y += (target_squash_y - self.squash_y) * 0.15

        now = time.time()
        curr_gpos = self.pos()
        d_gx = float(curr_gpos.x() - self.last_global_pos.x())
        d_gy = float(curr_gpos.y() - self.last_global_pos.y())
        self.last_global_pos = curr_gpos
        
        # 2. Liquid Drag Particles (Triggered by rapid dragging)
        drag_speed = math.hypot(d_gx, d_gy)
        if self.dragging and drag_speed > 3.0:
            num_particles = int(drag_speed * 0.7)
            trail_angle = math.atan2(-d_gy, -d_gx)
            for _ in range(num_particles):
                rx = 50.0 * scale * self.squash_x
                ry = 50.0 * scale * self.squash_y
                ang = trail_angle + random.uniform(-0.9, 0.9)
                dist = random.uniform(0.7, 1.1)
                
                px_x = self.last_head_cx + math.cos(ang) * rx * dist
                px_y = self.last_head_cy + math.sin(ang) * ry * dist
                
                self.particles.append({
                    'x': px_x,
                    'y': px_y,
                    'vx': -d_gx * random.uniform(5.0, 18.0) + random.uniform(-40, 40),
                    'vy': -d_gy * random.uniform(5.0, 18.0) + random.uniform(-40, 40),
                    'life': random.uniform(0.3, 0.65),
                    'size': random.choice([1, 1, 2, 2, 3])
                })

        for p in self.particles:
            p['x'] -= d_gx * 0.95
            p['y'] -= d_gy * 0.95
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['vy'] += 800.0 * dt
            p['life'] -= dt
            
        self.particles = [p for p in self.particles if p['life'] > 0]

        if self.dragging:
            self.drag_anim_val = min(1.0, self.drag_anim_val + 0.16)
            self.drag_phase += dt
            self.drag_vel_x += (d_gx - self.drag_vel_x) * 0.35
            self.drag_vel_y += (d_gy - self.drag_vel_y) * 0.35
        else:
            self.drag_anim_val = max(0.0, self.drag_anim_val - 0.08)
            self.drag_vel_x *= 0.8
            self.drag_vel_y *= 0.8

        if self.drag_anim_val > 0.01:
            drag_wobble = math.sin(self.drag_phase * 20.0) * 0.14
            drag_stretch = math.hypot(self.drag_vel_x, self.drag_vel_y) * 0.012
            drag_mouth_target = 0.74 + drag_wobble
            self.mouth_open = (1.0 - self.drag_anim_val) * self.mouth_open + self.drag_anim_val * drag_mouth_target
            if self.drag_anim_val > 0.25:
                self.viseme_mode = "O_PHONEME"
            self.brow_raise = (1.0 - self.drag_anim_val) * self.brow_raise + self.drag_anim_val * 0.88
            self.eye_openness = (1.0 - self.drag_anim_val) * self.eye_openness + self.drag_anim_val * (0.92 + drag_wobble * 0.5)
            target_drag_look_x = max(-1.0, min(1.0, self.drag_vel_x * 0.08)) + math.sin(self.drag_phase * 25.0) * 0.08 * self.drag_anim_val
            target_drag_look_y = max(-1.0, min(1.0, self.drag_vel_y * 0.08))
            self.look_x = (1.0 - self.drag_anim_val) * self.look_x + self.drag_anim_val * target_drag_look_x
            self.look_y = (1.0 - self.drag_anim_val) * self.look_y + self.drag_anim_val * target_drag_look_y
            self.mouth_shift_x -= self.drag_vel_x * 0.15 * self.drag_anim_val
            self.mouth_shift_y -= self.drag_vel_y * 0.15 * self.drag_anim_val
            if abs(self.drag_vel_y) > abs(self.drag_vel_x):
                self.squash_y = max(0.80, min(1.35, self.squash_y + drag_stretch * self.drag_anim_val))
                self.squash_x = max(0.80, min(1.25, 1.0 / self.squash_y))
            else:
                self.squash_x = max(0.80, min(1.35, self.squash_x + drag_stretch * self.drag_anim_val))
                self.squash_y = max(0.80, min(1.25, 1.0 / self.squash_x))

        d_hx_raw = td.get("head_x", 0.0) - self.head_x
        d_hy_raw = td.get("head_y", 0.0) - self.head_y
        h_vel_x = 0.0 if abs(d_hx_raw) < 0.015 else d_hx_raw * 25.0
        h_vel_y = 0.0 if abs(d_hy_raw) < 0.015 else d_hy_raw * 25.0
        inst_vel_x = d_gx * 6.5 + h_vel_x
        inst_vel_y = d_gy * 6.5 + h_vel_y

        self.cape_vel_x += (inst_vel_x - self.cape_vel_x) * 0.22
        self.cape_vel_y += (inst_vel_y - self.cape_vel_y) * 0.22
        if abs(self.cape_vel_x) < 0.05: self.cape_vel_x = 0.0
        if abs(self.cape_vel_y) < 0.05: self.cape_vel_y = 0.0

        self.cape_anim_state = "IDLE"
        anim_cape = 0.0
        if self.smile > 0.35: anim_cape += (self.smile - 0.35) * 0.25
        if self.is_flipping_x: anim_cape += math.sin(math.radians(self.rot_x)) * 0.6
        drag_torque = max(-1.0, min(1.0, -math.atan2(self.cape_vel_x * 0.035, 9.8) + anim_cape))
        
        k_cape = 32.0
        d_cape = 0.86
        force_cape = (drag_torque - self.cape_angle) * k_cape
        self.cape_avel = (self.cape_avel + force_cape * dt) * d_cape
        self.cape_angle += self.cape_avel * dt
        self.cape_curl += ((-self.cape_avel * 0.32) - self.cape_curl) * 0.18
        tgt_stretch = max(0.85, min(1.35, 1.0 + self.cape_vel_y * 0.015))
        self.cape_stretch += (tgt_stretch - self.cape_stretch) * 0.18

        px = max(2, int(round(3.0 * scale)))
        collar_w = max(px * 6, int(round(34.0 * scale)))
        hem_w = max(px * 14, int(round(74.0 * scale)))
        cape_h = max(px * 12, int(round(68.0 * scale)))

        w_f, h_f = float(self.width()), float(self.height())
        neck_anchor_x = w_f / 2.0 + self.neck_x * (20.0 * scale)
        neck_anchor_y = h_f / 2.0 + 32.0 * scale + self.neck_y * (18.0 * scale)
        cx_cur = neck_anchor_x + (self.head_x - self.neck_x * 0.75) * (26.0 * scale)
        cy_cur = neck_anchor_y - 28.0 * scale + (self.head_y - self.neck_y * 0.75) * (22.0 * scale)

        active_hat = self.hat_catalogue[self.active_hat_idx] if hasattr(self, "hat_catalogue") and self.hat_catalogue else None
        if self.show_cap and active_hat:
            cap_h_calc = active_hat["base_fit_h"] * scale
            hat_off_y_calc = active_hat.get("offset_y", 0.0) * scale
            safe_top_margin = max(58.0 * scale, (cap_h_calc - hat_off_y_calc + 36.0 * scale) + 12.0 * scale)
        else:
            safe_top_margin = 58.0 * scale

        safe_side_margin = 58.0 * scale
        cx_cur = max(safe_side_margin, min(w_f - safe_side_margin, cx_cur))
        cy_cur = max(safe_top_margin, min(h_f - 40.0 * scale, cy_cur))
        head_radius_y = 50.0 * scale * self.squash_y

        collar_cx = cx_cur + (self.yaw * 6.0 * scale)
        collar_cy = cy_cur + (head_radius_y * 0.15) + (self.pitch * 6.0 * scale)

        curr_collar_gx = float(curr_gpos.x() + collar_cx)
        curr_collar_gy = float(curr_gpos.y() + collar_cy)

        if not self.collar_initialized:
            self.last_collar_gx = curr_collar_gx
            self.last_collar_gy = curr_collar_gy
            self.collar_initialized = True

        d_collar_x = curr_collar_gx - self.last_collar_gx
        d_collar_y = curr_collar_gy - self.last_collar_gy
        self.last_collar_gx = curr_collar_gx
        self.last_collar_gy = curr_collar_gy

        diagonal_torsion = (self.yaw * self.pitch) * 16.0

        self.cape_sim.step(
            dt=dt, collar_w=collar_w, hem_w=hem_w, cape_h=cape_h, scale=scale,
            d_collar_x=d_collar_x, d_collar_y=d_collar_y, yaw=self.yaw, pitch=self.pitch,
            tilt_deg=self.tilt_deg + diagonal_torsion, anim_state=self.cape_anim_state,
            anim_timer=self.cape_anim_timer, smile=self.smile, is_flipping=self.is_flipping_x, rot_x=self.rot_x
        )

        self.last_collar_cx = collar_cx
        self.last_collar_cy = collar_cy

        if (self.show_pointers or self.show_inventory) and getattr(self, "_cg_event_btn", None) is not None:
            try:
                if self._cg_event_btn(0, 0) and not self.geometry().contains(QCursor.pos()):
                    self.show_pointers = False
                    self.show_inventory = False
            except Exception:
                pass

        if self.show_cap and self.cap_pixmap is not None and not self.cap_pixmap.isNull():
            if active_hat:
                cap_w = active_hat["base_fit_w"] * scale
                cap_h = active_hat["base_fit_h"] * scale
                hat_off_x = active_hat.get("offset_x", 0.0) * scale
                hat_off_y = active_hat.get("offset_y", 0.0) * scale
                if getattr(self, "hat_flipped", False): hat_off_x = -hat_off_x
            else:
                cap_w = 92.0 * scale
                cap_h = cap_w * (self.cap_pixmap.height() / float(self.cap_pixmap.width()))
                hat_off_x = 0.0
                hat_off_y = 0.0

            head_rx = 50.0 * scale * self.squash_x
            cranial_brow_y = cy_cur - head_radius_y * 0.72 + self.pitch * (head_radius_y * 0.35)
            target_cap_x = cx_cur - cap_w * 0.50 + self.yaw * (head_rx * 0.40) + hat_off_x
            target_cap_y = cranial_brow_y - cap_h + hat_off_y
            target_cap_rot = self.manual_rotation_z + self.tilt_deg * 0.85

            self.cap_instance.update(
                dt=dt, target_x=target_cap_x, target_y=target_cap_y, target_rot=target_cap_rot,
                drag_vx=d_gx, drag_vy=d_gy, scale=scale, is_dragging=self.dragging
            )

        eval_state = {
            "smile": self.smile, "mouth_open": self.mouth_open, "brow_raise": self.brow_raise,
            "tilt_deg": self.tilt_deg, "pitch": self.pitch, "yaw": self.yaw,
            "look_x": self.look_x, "look_y": self.look_y, "is_blinking": self.blink_val > 0.6,
            "velocity": td.get("velocity", 0.0), "is_spinning": self.is_flipping_x,
            "frown": td.get("frown", 0.0), "smile_asymmetry": td.get("smile_asymmetry", 0.0),
            "lip_pucker": self.lip_pucker, "eye_openness": self.eye_openness,
            "is_winking_left": td.get("is_winking_left", False), "is_winking_right": td.get("is_winking_right", False),
            "viseme_mode": self.viseme_mode, "upper_teeth": td.get("upper_teeth", 0.0),
            "lower_teeth": td.get("lower_teeth", 0.0), "mouth_shift_x": self.mouth_shift_x, "mouth_shift_y": self.mouth_shift_y,
        }
        self.current_mood = self.mood_engine.evaluate(eval_state)
        self.update()

    def create_3d_droplet_path(self, cx, cy, rx, ry, yaw=0.0, pitch=0.0):
        yaw_compression = max(0.68, 1.0 - abs(yaw) * 0.28)
        w_target = rx * 2.36 * yaw_compression
        h_target = ry * 2.36
        tip_yaw_dx = yaw * (rx * 0.48)
        tip_pitch_dy = pitch * (ry * 0.42)
        path = QPainterPath()
        tuck_ny = self.get_current_hat_tuck_ny()
        for i, (nx, ny) in enumerate(BASE_CONTOUR_NORM):
            eff_ny = tuck_ny if (self.show_cap and ny < tuck_ny) else ny
            if eff_ny < 0.0:
                w_tip = max(0.0, -eff_ny * 2.0) ** 1.3
                v_dx = tip_yaw_dx * w_tip
                v_dy = tip_pitch_dy * w_tip
            else:
                w_base = min(1.0, eff_ny * 2.0) ** 1.2
                v_dx = -tip_yaw_dx * 0.16 * w_base
                v_dy = -tip_pitch_dy * 0.14 * w_base
            persp_dx = yaw * (1.0 - abs(nx)) * (rx * 0.20)
            px = cx + nx * w_target + v_dx + persp_dx
            py = cy + eff_ny * h_target + 8.0 * (w_target / 240.0) + v_dy
            if i == 0: path.moveTo(px, py)
            else: path.lineTo(px, py)
        path.closeSubpath()
        return path

    def get_3d_side_shadow_paths(self, cx, cy, rx, ry, yaw, pitch, scale):
        front_path = self.create_3d_droplet_path(cx, cy, rx, ry, yaw, pitch)
        if abs(yaw) <= 0.10:
            return front_path, front_path, 0.0, 0.0
        side_w = -math.copysign(max(0.0, (abs(yaw) - 0.10) * 22.0 * scale), yaw)
        side_h = pitch * 6.0 * scale
        rear_path = self.create_3d_droplet_path(cx + side_w, cy + side_h, rx, ry, yaw, pitch)
        yaw_compression = max(0.68, 1.0 - abs(yaw) * 0.28)
        w_target = rx * 2.36 * yaw_compression
        h_target = ry * 2.36
        tip_yaw_dx = yaw * (rx * 0.48)
        tip_pitch_dy = pitch * (ry * 0.42)
        front_pts = []
        tuck_ny = self.get_current_hat_tuck_ny()
        for i, (nx, ny) in enumerate(BASE_CONTOUR_NORM):
            eff_ny = tuck_ny if (self.show_cap and ny < tuck_ny) else ny
            if eff_ny < 0.0:
                w_tip = max(0.0, -eff_ny * 2.0) ** 1.3
                v_dx = tip_yaw_dx * w_tip
                v_dy = tip_pitch_dy * w_tip
            else:
                w_base = min(1.0, eff_ny * 2.0) ** 1.2
                v_dx = -tip_yaw_dx * 0.16 * w_base
                v_dy = -tip_pitch_dy * 0.14 * w_base
            persp_dx = yaw * (1.0 - abs(nx)) * (rx * 0.20)
            px = cx + nx * w_target + v_dx + persp_dx
            py = cy + eff_ny * h_target + 8.0 * (w_target / 240.0) + v_dy
            front_pts.append((px, py, nx, ny))
        poly = QPolygonF()
        turning_right = (yaw > 0)
        flank_pts = [(px, py) for (px, py, nx, ny) in front_pts if (nx <= 0.05 if turning_right else nx >= -0.05)]
        flank_pts.sort(key=lambda p: p[1])
        for p in flank_pts: poly.append(QPointF(p[0], p[1]))
        for p in reversed(flank_pts): poly.append(QPointF(p[0] + side_w, p[1] + side_h))
        facet_path = QPainterPath()
        facet_path.addPolygon(poly)
        combined_path = front_path.united(rear_path).united(facet_path)
        return front_path, combined_path, side_w, side_h

    def draw_pixelated_body(self, painter, cx, cy, rx, ry, yaw, pitch, scale):
        px = max(2, int(round(2.5 * scale)))
        bw = int(rx * 3.6)
        bh = int(ry * 3.6)
        pw = max(64, bw // px)
        ph = max(64, bh // px)
        buf = QImage(pw, ph, QImage.Format.Format_ARGB32_Premultiplied)
        buf.fill(Qt.GlobalColor.transparent)
        bp = QPainter(buf)
        bp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        b_cx = pw / 2.0
        b_cy = ph / 2.0
        head_rx = rx / px
        head_ry = ry / px
        yaw_compression = max(0.68, 1.0 - abs(yaw) * 0.28)
        w_target = head_rx * 2.36 * yaw_compression
        h_target = head_ry * 2.36
        tip_yaw_dx = yaw * (head_rx * 0.48)
        tip_pitch_dy = pitch * (head_ry * 0.42)
        front_pts = []
        tuck_ny = self.get_current_hat_tuck_ny()
        for nx, ny in BASE_CONTOUR_NORM:
            eff_ny = tuck_ny if (self.show_cap and ny < tuck_ny) else ny
            if eff_ny < 0.0:
                w_tip = max(0.0, -eff_ny * 2.0) ** 1.3
                v_dx = tip_yaw_dx * w_tip
                v_dy = tip_pitch_dy * w_tip
            else:
                w_base = min(1.0, eff_ny * 2.0) ** 1.2
                v_dx = -tip_yaw_dx * 0.16 * w_base
                v_dy = -tip_pitch_dy * 0.14 * w_base
            persp_dx = yaw * (1.0 - abs(nx)) * (head_rx * 0.20)
            px_pt = b_cx + nx * w_target + v_dx + persp_dx
            py_pt = b_cy + eff_ny * h_target + 8.0 * (w_target / 240.0) + v_dy
            front_pts.append((px_pt, py_pt, nx, ny))
        front_poly = QPolygonF([QPointF(p[0], p[1]) for p in front_pts])
        has_shadow = abs(yaw) > 0.10 and getattr(self, "show_shadow", True)
        if has_shadow:
            side_w = -math.copysign(max(0.0, (abs(yaw) - 0.10) * 22.0 * scale / px), yaw)
            side_h = pitch * 6.0 * scale / px
            turning_right = (yaw > 0)
            flank_pts = [(p[0], p[1]) for p in front_pts if (p[2] <= 0.05 if turning_right else p[2] >= -0.05)]
            flank_pts.sort(key=lambda p: p[1])
            shadow_poly = QPolygonF()
            for p in flank_pts: shadow_poly.append(QPointF(p[0], p[1]))
            for p in reversed(flank_pts): shadow_poly.append(QPointF(p[0] + side_w, p[1] + side_h))

        mask_img = QImage(pw, ph, QImage.Format.Format_ARGB32_Premultiplied)
        mask_img.fill(Qt.GlobalColor.transparent)
        mp = QPainter(mask_img)
        mp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        mp.setPen(Qt.PenStyle.NoPen)
        mp.setBrush(QColor(255, 255, 255))
        if has_shadow: mp.drawPolygon(shadow_poly)
        mp.drawPolygon(front_poly)
        mp.end()

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
            bp.drawImage(dx, dy, mask_img)

        if has_shadow:
            bp.setPen(Qt.PenStyle.NoPen)
            shadow_pal = CAPE_PALETTES.get(getattr(self, "active_shadow_color", "red"), CAPE_PALETTES["red"])
            bp.setBrush(QColor(shadow_pal["dark"]))
            bp.drawPolygon(shadow_poly)

        bp.setPen(Qt.PenStyle.NoPen)
        bp.setBrush(QColor(self.theme.get("base_color", BLOSSOM_X6["BLACK"])))
        bp.drawPolygon(front_poly)
        bp.end()

        dest_w = pw * px
        dest_h = ph * px
        dest_rect = QRectF(cx - dest_w / 2.0, cy - dest_h / 2.0, dest_w, dest_h)
        prev_smooth = painter.renderHints() & QPainter.RenderHint.SmoothPixmapTransform
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.drawImage(dest_rect, buf)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, bool(prev_smooth))

    def draw_pixelated_cape(self, painter, anchor_x, anchor_y, head_rx, head_ry, scale):
        draw_pixelated_cape(painter, self.cape_sim, anchor_x, anchor_y, head_rx, head_ry, scale, getattr(self, "active_cape_color", "red"))

    def draw_pixelated_eye(self, painter, center_x, center_y, rx, ry, blink, scale, is_left=True):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        px = max(2, int(round(2.4 * scale)))
        overlap = px + 1.0  # Eliminates the dark grid/hairline gaps between pixels
        eye_color = QColor(self.theme["eye_color"])
        
        # Integrate manual click-to-blink
        manual_blink = math.sin(self.manual_blink_l_timer * math.pi) if is_left else math.sin(self.manual_blink_r_timer * math.pi)
        clean_blink = 0.0 if blink < 0.18 else (blink - 0.18) / 0.82
        clean_blink = max(clean_blink, manual_blink)

        if clean_blink > 0.52:
            bar_hw = max(px * 2, int(round(rx * 1.1 / px)) * px)
            for dx in range(-bar_hw, bar_hw + px, px):
                painter.fillRect(QRectF(center_x + dx, center_y - px / 2.0, overlap, overlap), eye_color)
            painter.restore()
            return

        eff_ry = max(px, ry * (1.0 - clean_blink * 0.88))
        eff_rx = max(px * 1.5, rx)
        max_gy = int(eff_ry) + px
        max_gx = int(eff_rx) + px
        for dy in range(-max_gy, max_gy + px, px):
            for dx in range(-max_gx, max_gx + px, px):
                nx = dx / eff_rx
                ny = dy / eff_ry
                if (nx * nx + ny * ny) <= 1.05:
                    painter.fillRect(QRectF(center_x + dx, center_y + dy, overlap, overlap), eye_color)

        if clean_blink < 0.35 and eff_ry > px * 2:
            sparkle_x = center_x - px * 1.5
            sparkle_y = center_y - px * 1.8
            painter.fillRect(QRectF(sparkle_x, sparkle_y, overlap, overlap), QColor(255, 255, 255, 250))
            
        painter.restore()

    def draw_pixelated_mouth(self, painter, mx, my, base_w, open_h, corner_lift, scale):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        px = max(2, int(round(2.4 * scale)))
        overlap = px + 1.0  # Eliminates the dark grid/hairline gaps between pixels
        
        c_white = QColor(BLOSSOM_X6["WHITE"])
        c_dark = QColor(BLOSSOM_X6["BLACK"])
        c_tongue = QColor(BLOSSOM_X6["CORAL"])

        is_resting = (self.mouth_open < 0.08 and self.smile < 0.28 and abs(self.smile_asymmetry) < 0.28 and self.frown < 0.28)
        if (is_resting and self.viseme_mode not in ("BILABIAL", "WIDE_GRIN", "SMIRK", "FROWN")) or self.viseme_mode == "NEUTRAL":
            seam_hw = max(px * 2, int(round(base_w * 0.38 / px)) * px)
            for bx in range(-seam_hw, seam_hw + px, px):
                painter.fillRect(QRectF(mx + bx, my, overlap, overlap), c_white)
            painter.restore()
            return

        is_frown = (self.frown > 0.28 or self.viseme_mode == "FROWN") and self.smile < 0.20
        is_smirk = (abs(self.smile_asymmetry) > 0.28 or self.viseme_mode == "SMIRK") and not is_frown
        is_triangle = (self.viseme_mode == "TRIANGLE_SMILE" or (self.smile > 0.40 and self.mouth_open > 0.10 and self.mouth_width >= 0.70 and abs(self.smile_asymmetry) < 0.25)) and self.theme_key == "base"
        is_wide_grin = (self.viseme_mode == "WIDE_GRIN" or (self.smile > 0.45 and self.mouth_open < 0.20 and not is_triangle and not is_smirk))
        is_pucker = (self.viseme_mode in ("PUCKER_U", "PUCKER") or (self.lip_pucker > 0.50 and self.mouth_width < 0.60)) and self.mouth_open < 0.12
        is_o = self.viseme_mode in ("ROUNDED_O", "O_PHONEME") or (self.lip_pucker > 0.38 and self.mouth_width < 0.68 and self.mouth_open >= 0.12)
        is_bilabial = self.viseme_mode == "BILABIAL"
        is_labiodental = self.viseme_mode == "LABIODENTAL"
        is_dental_alveolar = self.viseme_mode in ("DENTAL", "ALVEOLAR")
        is_postalveolar = self.viseme_mode == "POSTALVEOLAR"
        is_spread = self.viseme_mode == "SPREAD_E_I"
        is_open_vowel = self.viseme_mode in ("OPEN_VOWEL", "OPEN_JAW", "TALKING") or self.mouth_open >= 0.12

        if is_frown:
            seam_hw = max(px * 2, int(round(base_w * 0.40 / px)) * px)
            drop_px = max(px, int(round(max(self.frown, 0.35) * 5.0 * scale / px)) * px)
            for bx in range(-seam_hw, seam_hw + px, px):
                t = abs(bx) / float(max(1, seam_hw))
                step_y = my + int(round(t * t * drop_px / px)) * px
                painter.fillRect(QRectF(mx + bx, step_y, overlap, overlap), c_white)
        elif is_smirk:
            seam_hw = max(px * 2, int(round(base_w * 0.42 / px)) * px)
            asym = self.smile_asymmetry if abs(self.smile_asymmetry) > 0.05 else 0.4
            lift_px = max(px, int(round((max(self.smile, 0.25) * 6.0 + abs(asym) * 8.0) * scale / px)) * px)
            for bx in range(-seam_hw, seam_hw + px, px):
                t = (bx / float(seam_hw)) * math.copysign(1.0, asym)
                step_y = my - int(round(max(0.0, t) * lift_px / px)) * px
                painter.fillRect(QRectF(mx + bx, step_y, overlap, overlap), c_white)
        elif is_triangle:
            half_w = max(px * 3, int(round(base_w * 0.46 / px)) * px)
            tot_h = max(px * 4, int(round(open_h * 1.2 / px)) * px)
            num_t_rows = max(3, tot_h // px)
            top_y = my - int(round(corner_lift * 0.7 / px)) * px
            for r in range(num_t_rows):
                frac = r / float(max(1, num_t_rows - 1))
                r_hw = max(px, int(round((1.0 - frac) * half_w / px)) * px)
                cur_y = top_y + r * px
                painter.fillRect(QRectF(mx - r_hw - px, cur_y, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + r_hw, cur_y, overlap, overlap), c_white)
                if r == 0: painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_white)
                elif r == 1:
                    painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_white)
                    painter.fillRect(QRectF(mx, cur_y, overlap, overlap), c_dark)
                elif r >= num_t_rows - 2: painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_tongue)
                else: painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_dark)
            tip_y = top_y + num_t_rows * px
            painter.fillRect(QRectF(mx - px, tip_y, px * 2, overlap), c_white)
        elif is_wide_grin:
            half_w = max(px * 3, int(round(base_w * 0.48 / px)) * px)
            grin_h = max(px * 3, int(round(max(4.0 * scale, open_h * 0.65) / px)) * px)
            top_y = my - int(round(corner_lift * 0.8 / px)) * px
            num_g_rows = max(3, grin_h // px)
            for r in range(num_g_rows):
                cur_y = top_y + r * px
                frac = r / float(max(1, num_g_rows - 1))
                r_hw = max(px, int(round(half_w * (1.0 - frac * 0.35) / px)) * px)
                painter.fillRect(QRectF(mx - r_hw - px, cur_y, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + r_hw, cur_y, overlap, overlap), c_white)
                if r == 0 or r == 1:
                    painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_white)
                    painter.fillRect(QRectF(mx, cur_y, overlap, overlap), c_dark)
                else: painter.fillRect(QRectF(mx - r_hw, cur_y, r_hw * 2, overlap), c_dark)
            bot_y = top_y + num_g_rows * px
            for bx in range(-half_w // 2, half_w // 2 + px, px):
                painter.fillRect(QRectF(mx + bx, bot_y, overlap, overlap), c_white)
        elif is_bilabial:
            hw = max(px * 2, int(round(base_w * 0.40 / px)) * px)
            for bx in range(-hw, hw + px, px): painter.fillRect(QRectF(mx + bx, my, overlap, overlap), c_white)
            notch_w = max(px, int(round(hw * 0.50 / px)) * px)
            for bx in range(-notch_w, notch_w + px, px):
                painter.fillRect(QRectF(mx + bx, my - px, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + bx, my + px, overlap, overlap), c_white)
            painter.fillRect(QRectF(mx - notch_w, my, notch_w * 2, overlap), c_dark)
        elif is_labiodental:
            hw = max(px * 3, int(round(base_w * 0.44 / px)) * px)
            teeth_hw = max(px * 2, int(round(hw * 0.75 / px)) * px)
            painter.fillRect(QRectF(mx - teeth_hw, my - px * 2, teeth_hw * 2, px * 2 + 0.6), c_white)
            painter.fillRect(QRectF(mx - teeth_hw, my, teeth_hw * 2, overlap), c_dark)
            painter.fillRect(QRectF(mx - hw, my + px, hw * 2, overlap), c_white)
        elif is_dental_alveolar:
            hw = max(px * 3, int(round(base_w * 0.48 / px)) * px)
            painter.fillRect(QRectF(mx - hw, my - px * 2, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - hw, my - px, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - int(round(hw * 0.8)), my, int(round(hw * 1.6)), overlap), c_dark)
            painter.fillRect(QRectF(mx - hw, my + px, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - int(round(hw * 0.8)), my + px * 2, int(round(hw * 1.6)), overlap), c_white)
        elif is_postalveolar:
            hw = max(px * 3, int(round(base_w * 0.38 / px)) * px)
            hh = max(px * 2, int(round(max(open_h * 0.75, 7.0 * scale) / px)) * px)
            painter.fillRect(QRectF(mx - hw, my - hh, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - hw, my + hh, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - hw, my - hh, overlap, hh * 2), c_white)
            painter.fillRect(QRectF(mx + hw - px, my - hh, overlap, hh * 2), c_white)
            painter.fillRect(QRectF(mx - hw + px, my - hh + px, (hw - px) * 2, (hh - px) * 2), c_dark)
            painter.fillRect(QRectF(mx - px * 2, my + hh - px * 2, px * 4, overlap), c_tongue)
        elif is_o:
            ow = max(px * 4, int(round(base_w * 0.38 / px)) * px)
            oh = max(px * 5, int(round(max(open_h * 1.1, 8.0 * scale) / px)) * px)
            half_ow = ow // 2
            half_oh = oh // 2
            for by in range(-half_oh, half_oh + px, px):
                for bx in range(-half_ow, half_ow + px, px):
                    dist_sq = (bx / float(max(1, half_ow))) ** 2 + (by / float(max(1, half_oh))) ** 2
                    if 0.52 <= dist_sq <= 1.25: painter.fillRect(QRectF(mx + bx, my + by, overlap, overlap), c_white)
                    elif dist_sq < 0.52:
                        if by > 0 and abs(bx) < half_ow * 0.45: painter.fillRect(QRectF(mx + bx, my + by, overlap, overlap), c_tongue)
                        else: painter.fillRect(QRectF(mx + bx, my + by, overlap, overlap), c_dark)
        elif is_pucker:
            pw = max(px * 3, int(round(base_w * 0.26 / px)) * px)
            ph = max(px * 3, int(round(max(open_h * 0.60, 7.0 * scale) / px)) * px)
            for bx in range(-pw // 2, pw // 2 + px, px):
                painter.fillRect(QRectF(mx + bx, my - ph // 2, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + bx, my + ph // 2, overlap, overlap), c_white)
            for by in range(-ph // 2, ph // 2 + px, px):
                painter.fillRect(QRectF(mx - pw // 2, my + by, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + pw // 2, my + by, overlap, overlap), c_white)
            painter.fillRect(QRectF(mx - pw // 2 + px, my - ph // 2 + px, pw - px, ph - px), c_dark)
        elif is_spread:
            hw = max(px * 3, int(round(base_w * 0.52 / px)) * px)
            tot_h = max(px * 3, int(round(max(open_h * 0.70, 7.0 * scale) / px)) * px)
            top_y = my - tot_h // 2
            painter.fillRect(QRectF(mx - hw, top_y, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - hw, top_y + px, hw * 2, overlap), c_white)
            painter.fillRect(QRectF(mx - int(round(hw * 0.8)), top_y + px * 2, int(round(hw * 1.6)), overlap), c_tongue)
            painter.fillRect(QRectF(mx - hw, top_y + px * 3, hw * 2, overlap), c_white)
        elif is_open_vowel:
            hw = max(px * 3, int(round(base_w * 0.48 / px)) * px)
            jaw_h = max(px * 4, int(round(max(open_h, 8.0 * scale) / px)) * px)
            num_j_rows = max(4, jaw_h // px)
            top_y = my - int(round(corner_lift * 0.8 / px)) * px - jaw_h // 4
            for r in range(num_j_rows):
                cur_y = top_y + r * px
                frac = r / float(max(1, num_j_rows - 1))
                cur_hw = max(px, int(round(hw * (1.0 - abs(frac - 0.5) * 0.28) / px)) * px)
                painter.fillRect(QRectF(mx - cur_hw - px, cur_y, overlap, overlap), c_white)
                painter.fillRect(QRectF(mx + cur_hw, cur_y, overlap, overlap), c_white)
                if r == 0: painter.fillRect(QRectF(mx - cur_hw, cur_y, cur_hw * 2, overlap), c_white)
                elif r == 1:
                    painter.fillRect(QRectF(mx - cur_hw, cur_y, cur_hw * 2, overlap), c_white)
                    painter.fillRect(QRectF(mx, cur_y, overlap, overlap), c_dark)
                elif r == num_j_rows - 2 and (self.lower_teeth > 0.12 or self.mouth_open > 0.45):
                    painter.fillRect(QRectF(mx - cur_hw * 0.6, cur_y, cur_hw * 1.2, overlap), c_white)
                elif r >= num_j_rows - 2: painter.fillRect(QRectF(mx - cur_hw * 0.5, cur_y, cur_hw, overlap), c_tongue)
                else: painter.fillRect(QRectF(mx - cur_hw, cur_y, cur_hw * 2, overlap), c_dark)
            bot_y = top_y + num_j_rows * px
            for bx in range(-hw // 2, hw // 2 + px, px): painter.fillRect(QRectF(mx + bx, bot_y, overlap, overlap), c_white)
        else:
            seam_hw = max(px * 2, int(round(base_w * 0.38 / px)) * px)
            for bx in range(-seam_hw, seam_hw + px, px): painter.fillRect(QRectF(mx + bx, my, overlap, overlap), c_white)
            
        painter.restore()

    def update_click_mask(self):
        self.clearMask()
        if self.click_through_mode:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    def showEvent(self, event):
        super().showEvent(event)
        make_permanent_always_on_top(self)
        self.update_click_mask()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 1))

        # Dynamic depth scaling amplification applied visually
        dynamic_depth_scale = max(0.4, min(1.8, 0.5 + self.depth * 0.5))
        scale = (min(self.width(), self.height()) / 280.0) * dynamic_depth_scale
        w, h = float(self.width()), float(self.height())

        neck_anchor_x = w / 2.0 + self.neck_x * (20.0 * scale)
        neck_anchor_y = h / 2.0 + 32.0 * scale + self.neck_y * (18.0 * scale)

        cx = neck_anchor_x + (self.head_x - self.neck_x * 0.75) * (26.0 * scale)
        cy = neck_anchor_y - 28.0 * scale + (self.head_y - self.neck_y * 0.75) * (22.0 * scale)

        active_hat = self.hat_catalogue[self.active_hat_idx] if hasattr(self, "hat_catalogue") and self.hat_catalogue else None
        if self.show_cap and active_hat:
            cap_h_calc = active_hat["base_fit_h"] * scale
            hat_off_y_calc = active_hat.get("offset_y", 0.0) * scale
            safe_top_margin = max(58.0 * scale, (cap_h_calc - hat_off_y_calc + 36.0 * scale) + 12.0 * scale)
        else:
            safe_top_margin = 58.0 * scale

        safe_side_margin = 58.0 * scale
        cx = max(safe_side_margin, min(w - safe_side_margin, cx))
        cy = max(safe_top_margin, min(h - 40.0 * scale, cy))

        head_radius_x = 50.0 * scale * self.squash_x
        head_radius_y = 50.0 * scale * self.squash_y

        mood_info = self.current_mood.get("info", {})
        gloss_mult = mood_info.get("gloss_boost", 1.0)
        pupil_mult = mood_info.get("pupil_scale", 1.0)
        glow_alpha = mood_info.get("glow_alpha", 45)

        diagonal_torsion = (self.yaw * self.pitch) * 16.0
        total_rot = (self.manual_rotation_z + self.tilt_deg + diagonal_torsion) % 360.0

        self.last_head_cx = cx
        self.last_head_cy = cy
        self.last_head_rx = head_radius_x
        self.last_head_ry = head_radius_y
        self.last_head_rot = total_rot

        collar_cx = cx + (self.yaw * 6.0 * scale)
        collar_cy = cy + (head_radius_y * 0.15) + (self.pitch * 6.0 * scale)

        if getattr(self, "show_cape", True):
            if self.manual_rotation_z != 0.0:
                painter.save()
                painter.translate(cx, cy)
                painter.rotate(self.manual_rotation_z)
                painter.translate(-cx, -cy)
                self.draw_pixelated_cape(painter, collar_cx, collar_cy, head_radius_x, head_radius_y, scale)
                painter.restore()
            else:
                self.draw_pixelated_cape(painter, collar_cx, collar_cy, head_radius_x, head_radius_y, scale)

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(total_rot)

        flip_angle_rad = math.radians(self.rot_x)
        cos_x = math.cos(flip_angle_rad)
        scale_y = math.copysign(max(0.06, abs(cos_x)), cos_x)

        painter.scale(1.0, scale_y)
        painter.translate(-cx, -cy)

        front_path, combined_path, side_w, side_h = self.get_3d_side_shadow_paths(cx, cy, head_radius_x, head_radius_y, self.yaw, self.pitch, scale)
        droplet_path = front_path

        if self.theme_key == "base":
            self.draw_pixelated_body(painter, cx, cy, head_radius_x, head_radius_y, self.yaw, self.pitch, scale)
        else:
            glow_color = QColor(self.theme["rim_color"])
            glow_color.setAlpha(int(glow_alpha))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(glow_color, 8.0 * scale))
            painter.drawPath(droplet_path)

            light_focal_x = cx - head_radius_x * 0.32 - self.yaw * (head_radius_x * 0.40)
            light_focal_y = cy - head_radius_y * 0.28 + self.pitch * (head_radius_y * 0.35)

            sphere_grad = QRadialGradient(light_focal_x, light_focal_y, head_radius_x * 1.38, light_focal_x, light_focal_y)
            c_highlight = QColor(self.theme["highlight_color"])
            c_highlight.setAlpha(240)
            c_base = QColor(self.theme["base_color"])
            c_deep = QColor(self.theme["deep_color"])

            sphere_grad.setColorAt(0.0, c_highlight)
            sphere_grad.setColorAt(0.28, c_base)
            sphere_grad.setColorAt(0.82, c_deep)
            sphere_grad.setColorAt(1.0, QColor(BLOSSOM_X6["BLACK"]))

            painter.setBrush(QBrush(sphere_grad))
            painter.setPen(QPen(QColor(self.theme["rim_color"]), 2.8 * scale))
            painter.drawPath(droplet_path)

            spec_x = light_focal_x - 4.0 * scale
            spec_y = light_focal_y - 6.0 * scale
            spec_grad = QRadialGradient(spec_x, spec_y, 22.0 * scale * gloss_mult)
            spec_grad.setColorAt(0.0, QColor(255, 255, 255, int(230 * min(1.0, gloss_mult))))
            spec_grad.setColorAt(0.45, QColor(255, 255, 255, int(110 * min(1.0, gloss_mult))))
            spec_grad.setColorAt(1.0, QColor(255, 255, 255, 0))

            painter.setBrush(QBrush(spec_grad))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRectF(spec_x - 14 * scale, spec_y - 12 * scale, 28 * scale * gloss_mult, 24 * scale * gloss_mult))

            painter.save()
            rim_path = QPainterPath()
            tip_x = cx + self.yaw * 14.0 * scale
            tip_y = cy - head_radius_y * 1.32 + self.pitch * 16.0 * scale
            rim_path.moveTo(tip_x - 4.0 * scale, tip_y + 14.0 * scale)
            rim_path.cubicTo(
                cx - head_radius_x * 0.70, cy - head_radius_y * 0.35,
                cx - head_radius_x * 0.82, cy + head_radius_y * 0.20,
                cx - head_radius_x * 0.50, cy + head_radius_y * 0.75
            )
            rim_color = QColor(255, 255, 255, int(75 * gloss_mult))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(rim_color, 2.5 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(rim_path)
            painter.restore()

        face_visible = cos_x > 0.0

        if face_visible:
            yaw_rad = self.yaw * 0.65
            pitch_rad = self.pitch * 0.55

            if self.theme_key == "base":
                eye_socket_y = cy + 10.0 * scale + (pitch_rad * 12.0 * scale)
            else:
                eye_socket_y = cy - 2.0 * scale + (pitch_rad * 14.0 * scale)

            left_phi = -0.42 + yaw_rad
            right_phi = 0.42 + yaw_rad

            eye_socket_lx = cx + math.sin(left_phi) * (head_radius_x * 0.72)
            eye_socket_rx = cx + math.sin(right_phi) * (head_radius_x * 0.72)

            gaze_shift_x = self.look_x * 7.5 * scale
            gaze_shift_y = self.look_y * 6.5 * scale

            eye_left_x = eye_socket_lx + gaze_shift_x
            eye_right_x = eye_socket_rx + gaze_shift_x
            eye_base_y = eye_socket_y + gaze_shift_y

            # Save un-projected center for hit-testing manual blinks
            self.last_eye_l = QPointF(eye_left_x, eye_base_y)
            self.last_eye_r = QPointF(eye_right_x, eye_base_y)

            brow_scale_y = max(0.35, min(1.55, self.eye_openness * 1.5))
            base_eye_r = (6.4 if self.theme_key == "base" else 5.6) * scale * pupil_mult

            eye_l_scale_x = max(0.45, math.cos(left_phi))
            eye_r_scale_x = max(0.45, math.cos(right_phi))

            left_blink = self.blink_left
            right_blink = self.blink_right

            self.draw_pixelated_eye(painter, eye_left_x, eye_base_y, base_eye_r * eye_l_scale_x, base_eye_r * brow_scale_y, left_blink, scale, is_left=True)
            self.draw_pixelated_eye(painter, eye_right_x, eye_base_y, base_eye_r * eye_r_scale_x, base_eye_r * brow_scale_y, right_blink, scale, is_left=False)

            mouth_phi = yaw_rad * 0.8
            mx = cx + math.sin(mouth_phi) * (head_radius_x * 0.58) + (self.mouth_shift_x * 10.0 * scale)
            if self.theme_key == "base":
                my = cy + 34.0 * scale + (pitch_rad * 10.0 * scale) + (self.mouth_shift_y * 8.0 * scale)
            else:
                my = cy + 22.0 * scale + (pitch_rad * 10.0 * scale) + (self.mouth_shift_y * 8.0 * scale)

            base_w = (26.0 + (self.mouth_width - 0.75) * 22.0 + self.smile * 14.0 - self.lip_pucker * 10.0) * scale * max(0.5, math.cos(mouth_phi))
            open_h = (self.mouth_open * 28.0 + self.smile * 8.0) * scale
            corner_lift = (self.smile * 6.5 - self.lip_pucker * 2.0) * scale

            self.draw_pixelated_mouth(painter, mx, my, base_w, open_h, corner_lift, scale)

        painter.restore()

        if self.particles:
            painter.save()
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(Qt.PenStyle.NoPen)
            particle_color = QColor(self.theme.get("base_color", BLOSSOM_X6["BLACK"]))
            painter.setBrush(particle_color)
            
            px_block = max(2, int(round(2.5 * scale)))
            
            for p in self.particles:
                gx = round(p['x'] / px_block) * px_block
                gy = round(p['y'] / px_block) * px_block
                s = px_block * p['size']
                if p['life'] < 0.15 and p['size'] > 1:
                    s -= px_block
                if s > 0:
                    painter.fillRect(QRectF(gx, gy, s, s), particle_color)
            painter.restore()

        if self.show_cap and self.cap_pixmap is not None and not self.cap_pixmap.isNull():
            active_hat = self.hat_catalogue[self.active_hat_idx] if hasattr(self, "hat_catalogue") and self.hat_catalogue else None
            if active_hat:
                cap_w = active_hat["base_fit_w"] * scale
                cap_h = active_hat["base_fit_h"] * scale
                hat_off_x = active_hat.get("offset_x", 0.0) * scale
                hat_off_y = active_hat.get("offset_y", 0.0) * scale
                if getattr(self, "hat_flipped", False): hat_off_x = -hat_off_x
            else:
                cap_w = 92.0 * scale
                cap_h = cap_w * (self.cap_pixmap.height() / float(self.cap_pixmap.width()))
                hat_off_x = 0.0
                hat_off_y = 0.0

            cap_x = self.cap_instance.x
            cap_y = self.cap_instance.y
            cap_rot = self.cap_instance.rot

            painter.save()
            painter.translate(cap_x + cap_w / 2.0, cap_y + cap_h / 2.0)
            painter.rotate(cap_rot)
            
            if getattr(self, "hat_flipped", False):
                painter.scale(-1.0, 1.0)
                
            painter.translate(-cap_w / 2.0, -cap_h / 2.0)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

            if abs(self.yaw) > 0.10:
                hat_side_w = -math.copysign(max(0.0, (abs(self.yaw) - 0.10) * 14.0 * scale), self.yaw)
                if getattr(self, "hat_flipped", False):
                    hat_side_w = -hat_side_w
                hat_side_h = self.pitch * 4.0 * scale
                if hasattr(self, "cap_silhouette") and self.cap_silhouette is not None:
                    colored_shadow = self.get_colored_hat_shadow()
                    for bdx, bdy in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)]:
                        painter.drawPixmap(QRectF(hat_side_w + bdx, hat_side_h + bdy, cap_w, cap_h), self.cap_silhouette, QRectF(0, 0, self.cap_silhouette.width(), self.cap_silhouette.height()))
                    painter.drawPixmap(QRectF(hat_side_w, hat_side_h, cap_w, cap_h), colored_shadow, QRectF(0, 0, colored_shadow.width(), colored_shadow.height()))

            if hasattr(self, "cap_silhouette") and self.cap_silhouette is not None and not self.cap_silhouette.isNull():
                b_px = 1.0
                for dx, dy in [(-b_px, 0.0), (b_px, 0.0), (0.0, -b_px), (0.0, b_px), (-b_px, -b_px), (b_px, -b_px), (-b_px, b_px), (b_px, b_px)]:
                    painter.drawPixmap(QRectF(dx, dy, cap_w, cap_h), self.cap_silhouette, QRectF(0, 0, self.cap_silhouette.width(), self.cap_silhouette.height()))

            painter.drawPixmap(QRectF(0, 0, cap_w, cap_h), self.cap_pixmap, QRectF(0, 0, self.cap_pixmap.width(), self.cap_pixmap.height()))
            painter.restore()

        if self.show_inventory:
            self.draw_hat_inventory(painter, scale)

        self.draw_pointer_handles(painter)

    def is_in_resize_corner(self, pos):
        return pos.x() >= self.width() - self.resize_corner_margin and pos.y() >= self.height() - self.resize_corner_margin

    def rotate_by(self, delta_deg):
        self.manual_rotation_z = (self.manual_rotation_z + delta_deg) % 360.0
        self.update()
        self.update_click_mask()

    def reset_rotation(self):
        self.manual_rotation_z = 0.0
        self.update()
        self.update_click_mask()

    def get_handle_positions(self):
        cx = getattr(self, "last_head_cx", self.width() / 2.0)
        cy = getattr(self, "last_head_cy", self.height() / 2.0)
        rx = getattr(self, "last_head_rx", 50.0 * (self.width() / 280.0))
        ry = getattr(self, "last_head_ry", 50.0 * (self.height() / 280.0))
        rot = getattr(self, "last_head_rot", self.manual_rotation_z)
        scale = (min(self.width(), self.height()) / 280.0) * self.depth

        theta = math.radians(rot)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        top_dx = 0.0
        top_dy = -ry * 1.38
        top_x = cx + top_dx * cos_t - top_dy * sin_t
        top_y = cy + top_dx * sin_t + top_dy * cos_t

        r_dx = rx * 1.36
        r_dy = 0.0
        r_x = cx + r_dx * cos_t - r_dy * sin_t
        r_y = cy + r_dx * sin_t + r_dy * cos_t

        if hasattr(self, "hat_catalogue") and self.hat_catalogue and self.active_hat_idx < len(self.hat_catalogue):
            active_hat = self.hat_catalogue[self.active_hat_idx]
            hat_w = active_hat["base_fit_w"] * scale
            hat_h = active_hat["base_fit_h"] * scale
            hat_cx = self.cap_instance.x + hat_w * 0.5
            hat_cy = self.cap_instance.y + hat_h * 0.5
            rad_hat = math.radians(self.cap_instance.rot)
            cos_h = math.cos(rad_hat)
            sin_h = math.sin(rad_hat)
            local_px = hat_w * 0.28
            local_py = -hat_h * 0.28
            p_x = hat_cx + local_px * cos_h - local_py * sin_h
            p_y = hat_cy + local_px * sin_h + local_py * cos_h
        else:
            p_x = cx + rx * 0.6 * cos_t - (-ry * 1.1) * sin_t
            p_y = cy + rx * 0.6 * sin_t + (-ry * 1.1) * cos_t

        margin = 14.0
        top_x = max(margin, min(self.width() - margin, top_x))
        top_y = max(margin, min(self.height() - margin, top_y))
        r_x = max(margin, min(self.width() - margin, r_x))
        r_y = max(margin, min(self.height() - margin, r_y))
        p_x = max(margin, min(self.width() - margin, p_x))
        p_y = max(margin, min(self.height() - margin, p_y))

        return (top_x, top_y), (r_x, r_y), (p_x, p_y)

    def hit_test_handles(self, pos):
        if not self.show_pointers: return None
        top_pos, right_pos, pencil_pos = self.get_handle_positions()
        scale = self.width() / 220.0
        hit_radius = max(14.0, 16.0 * scale)

        if math.hypot(pos.x() - pencil_pos[0], pos.y() - pencil_pos[1]) <= hit_radius: return "PENCIL"
        if math.hypot(pos.x() - top_pos[0], pos.y() - top_pos[1]) <= hit_radius: return "TOP"
        if math.hypot(pos.x() - right_pos[0], pos.y() - right_pos[1]) <= hit_radius: return "RIGHT"
        return None

    def draw_pointer_handles(self, painter):
        if not self.show_pointers: return
        top_pos, right_pos, pencil_pos = self.get_handle_positions()
        scale = self.width() / 220.0
        handle_r = max(7.5, 9.0 * scale)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        cx = getattr(self, "last_head_cx", self.width() / 2.0)
        cy = getattr(self, "last_head_cy", self.height() / 2.0)

        c_white = QColor(BLOSSOM_X6["WHITE"])
        c_dark = QColor(BLOSSOM_X6["BLACK"])
        c_scale = QColor(BLOSSOM_X6["CRIMSON"])
        c_rotate = QColor(BLOSSOM_X6["CORAL"])
        c_pencil = QColor(BLOSSOM_X6["CRIMSON"])

        stem_pen = QPen(QColor(255, 255, 255, 130), 1.5, Qt.PenStyle.DashLine)
        painter.setPen(stem_pen)
        painter.drawLine(QPointF(cx, cy), QPointF(top_pos[0], top_pos[1]))
        painter.drawLine(QPointF(cx, cy), QPointF(right_pos[0], right_pos[1]))
        painter.drawLine(QPointF(cx, cy), QPointF(pencil_pos[0], pencil_pos[1]))

        tx, ty = top_pos
        if self.hover_top_handle or self.scaling_active:
            painter.setBrush(QBrush(QColor(255, 75, 75, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(tx, ty), handle_r + 4.5, handle_r + 4.5)
        painter.setBrush(QBrush(c_dark))
        painter.setPen(QPen(c_white, 2.0))
        painter.drawEllipse(QPointF(tx, ty), handle_r, handle_r)
        inner_r = handle_r * 0.55
        painter.setBrush(QBrush(c_scale))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(tx, ty), inner_r, inner_r)
        arrow_pen = QPen(c_white, 1.5)
        painter.setPen(arrow_pen)
        painter.drawLine(QPointF(tx, ty - inner_r * 0.7), QPointF(tx, ty + inner_r * 0.7))
        painter.drawLine(QPointF(tx - 2.5, ty - inner_r * 0.35), QPointF(tx, ty - inner_r * 0.7))
        painter.drawLine(QPointF(tx + 2.5, ty - inner_r * 0.35), QPointF(tx, ty - inner_r * 0.7))
        painter.drawLine(QPointF(tx - 2.5, ty + inner_r * 0.35), QPointF(tx, ty + inner_r * 0.7))
        painter.drawLine(QPointF(tx + 2.5, ty + inner_r * 0.35), QPointF(tx, ty + inner_r * 0.7))

        rx, ry = right_pos
        if self.hover_right_handle or self.handle_rotating:
            painter.setBrush(QBrush(QColor(255, 75, 75, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(rx, ry), handle_r + 4.5, handle_r + 4.5)
        painter.setBrush(QBrush(c_dark))
        painter.setPen(QPen(c_white, 2.0))
        painter.drawEllipse(QPointF(rx, ry), handle_r, handle_r)
        painter.setBrush(QBrush(c_rotate))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(rx, ry), inner_r, inner_r)
        rot_pen = QPen(c_white, 1.4)
        painter.setPen(rot_pen)
        painter.drawArc(QRectF(rx - inner_r * 0.65, ry - inner_r * 0.65, inner_r * 1.3, inner_r * 1.3), 45 * 16, 270 * 16)

        px_x, px_y = pencil_pos
        if getattr(self, "hover_pencil_handle", False) or self.show_inventory:
            painter.setBrush(QBrush(QColor(255, 75, 75, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px_x, px_y), handle_r + 4.5, handle_r + 4.5)
        painter.setBrush(QBrush(c_dark))
        painter.setPen(QPen(c_white, 2.0))
        painter.drawEllipse(QPointF(px_x, px_y), handle_r, handle_r)
        painter.setBrush(QBrush(c_pencil))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(px_x, px_y), inner_r, inner_r)

        painter.save()
        painter.translate(px_x, px_y)
        painter.rotate(-45)
        sw = max(2.5, 3.2 * (scale / 1.2))
        sh = max(5.5, 7.5 * (scale / 1.2))
        painter.fillRect(QRectF(-sw / 2.0, -sh / 2.0 + 1.0, sw, sh - 2.0), c_white)
        painter.fillRect(QRectF(-sw / 2.0, -sh / 2.0 - 1.2, sw, 2.0), QColor(BLOSSOM_X6["RUBY"]))
        tip_poly = QPolygonF([QPointF(-sw / 2.0, sh / 2.0 - 1.0), QPointF(sw / 2.0, sh / 2.0 - 1.0), QPointF(0.0, sh / 2.0 + 2.8)])
        painter.setBrush(QBrush(c_white))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(tip_poly)
        lead_poly = QPolygonF([QPointF(-sw * 0.25, sh / 2.0 + 1.0), QPointF(sw * 0.25, sh / 2.0 + 1.0), QPointF(0.0, sh / 2.0 + 2.8)])
        painter.setBrush(QBrush(c_dark))
        painter.drawPolygon(lead_poly)
        painter.restore()
        painter.restore()

    def hit_test_inventory(self, pos):
        if not self.show_inventory: return None
        p = QPointF(pos)
        for val, rect in getattr(self, "inventory_hat_slot_rects", []):
            if rect.contains(p): return ("HAT", val)
        for cid, rect in getattr(self, "inventory_cape_slot_rects", []):
            if rect.contains(p): return ("CAPE", cid)
        for sid, rect in getattr(self, "inventory_shadow_slot_rects", []):
            if rect.contains(p): return ("SHADOW", sid)
        return None

    def is_inside_inventory(self, pos):
        if not self.show_inventory or self.inventory_rect.isNull(): return False
        return self.inventory_rect.contains(QPointF(pos))

    def draw_hat_inventory(self, painter, scale):
        if not hasattr(self, "hat_catalogue") or not self.hat_catalogue: return
        N_hats = len(self.hat_catalogue)
        item_keys = ["red", "blue", "green", "purple"]
        N_cols = max(N_hats, len(item_keys))
        slot_w = max(34.0, 40.0 * scale)
        slot_h_hat = max(32.0, 36.0 * scale)
        slot_h_cape = max(18.0, 22.0 * scale)
        slot_gap = max(4.0, 6.0 * scale)
        row_gap = max(4.0, 6.0 * scale)
        pad_x = max(6.0, 8.0 * scale)
        pad_y = max(6.0, 8.0 * scale)

        inv_w = N_cols * slot_w + (N_cols - 1) * slot_gap + pad_x * 2.0
        inv_h = pad_y + slot_h_hat + row_gap + slot_h_cape + row_gap + slot_h_cape + pad_y

        collar_cy = getattr(self, "last_collar_cy", self.height() * 0.55)
        target_y = collar_cy + (70.0 * scale)
        if target_y + inv_h > self.height() - 6.0:
            target_y = self.height() - inv_h - 6.0
        target_x = (self.width() - inv_w) / 2.0

        self.inventory_rect = QRectF(target_x, target_y, inv_w, inv_h)
        self.inventory_hat_slot_rects = []
        self.inventory_cape_slot_rects = []
        self.inventory_shadow_slot_rects = []

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        panel_color = QColor(10, 10, 15, 238)
        border_pen = QPen(QColor(BLOSSOM_X6["DEEP_BERRY"]), max(1.5, 2.0 * scale))
        painter.setBrush(QBrush(panel_color))
        painter.setPen(border_pen)
        painter.drawRoundedRect(self.inventory_rect, 8.0 * scale, 8.0 * scale)

        painter.setPen(QPen(QColor(255, 255, 255, 120), 1.0))
        painter.drawLine(QPointF(target_x + 8.0 * scale, target_y + 1.0), QPointF(target_x + inv_w - 8.0 * scale, target_y + 1.0))
        hover_slot = getattr(self, "hover_inventory_slot", None)

        for i, hat in enumerate(self.hat_catalogue):
            sx = target_x + pad_x + i * (slot_w + slot_gap)
            sy = target_y + pad_y
            s_rect = QRectF(sx, sy, slot_w, slot_h_hat)
            self.inventory_hat_slot_rects.append((i, s_rect))

            is_active = (self.show_cap and i == self.active_hat_idx)
            is_hover = (hover_slot == ("HAT", i))

            if is_active:
                painter.setBrush(QBrush(QColor(179, 0, 60, 160)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["CORAL"]), max(1.8, 2.2 * scale)))
            elif is_hover:
                painter.setBrush(QBrush(QColor(255, 255, 255, 35)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["WHITE"]), 1.5))
            else:
                painter.setBrush(QBrush(QColor(22, 22, 28, 200)))
                painter.setPen(QPen(QColor(80, 80, 95, 120), 1.0))

            painter.drawRoundedRect(s_rect, 5.0 * scale, 5.0 * scale)
            pm = hat["pixmap"]
            sil = hat["white_sil"]
            avail_w = slot_w - 8.0 * scale
            avail_h = slot_h_hat - 8.0 * scale
            aspect = pm.width() / float(max(1, pm.height()))
            if aspect >= 1.0:
                t_w = avail_w
                t_h = t_w / aspect
                if t_h > avail_h:
                    t_h = avail_h
                    t_w = t_h * aspect
            else:
                t_h = avail_h
                t_w = t_h * aspect
                if t_w > avail_w:
                    t_w = avail_w
                    t_h = t_w / aspect
            t_x = sx + (slot_w - t_w) / 2.0
            t_y = sy + (slot_h_hat - t_h) / 2.0
            t_rect = QRectF(t_x, t_y, t_w, t_h)

            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            for bdx, bdy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                painter.drawPixmap(t_rect.translated(bdx, bdy), sil, QRectF(0, 0, sil.width(), sil.height()))
            painter.drawPixmap(t_rect, pm, QRectF(0, 0, pm.width(), pm.height()))

            if is_active:
                pip_r = max(2.5, 3.2 * scale)
                painter.setBrush(QBrush(QColor(BLOSSOM_X6["WHITE"])))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(sx + slot_w - pip_r * 1.5, sy + pip_r * 1.5), pip_r, pip_r)

        div_y = target_y + pad_y + slot_h_hat + (row_gap / 2.0)
        painter.setPen(QPen(QColor(BLOSSOM_X6["DEEP_BERRY"]), 1.0))
        painter.drawLine(QPointF(target_x + pad_x + 4.0, div_y), QPointF(target_x + inv_w - pad_x - 4.0, div_y))

        cape_sy = target_y + pad_y + slot_h_hat + row_gap
        for j, ckey in enumerate(item_keys):
            csx = target_x + pad_x + j * (slot_w + slot_gap)
            c_rect = QRectF(csx, cape_sy, slot_w, slot_h_cape)
            self.inventory_cape_slot_rects.append((ckey, c_rect))

            is_active_cape = (self.show_cape and ckey == getattr(self, "active_cape_color", "red"))
            is_hover_cape = (hover_slot == ("CAPE", ckey))

            if is_active_cape:
                painter.setBrush(QBrush(QColor(179, 0, 60, 160)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["WHITE"]), max(1.5, 1.8 * scale)))
            elif is_hover_cape:
                painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["WHITE"]), 1.4))
            else:
                painter.setBrush(QBrush(QColor(18, 18, 24, 210)))
                painter.setPen(QPen(QColor(60, 60, 75, 120), 1.0))

            painter.drawRoundedRect(c_rect, 4.0 * scale, 4.0 * scale)

            pal = CAPE_PALETTES.get(ckey, CAPE_PALETTES["red"])
            swatch_w = max(22.0, 26.0 * scale)
            swatch_h = max(10.0, 12.0 * scale)
            sw_x = csx + (slot_w - swatch_w) / 2.0
            sw_y = cape_sy + (slot_h_cape - swatch_h) / 2.0
            painter.fillRect(QRectF(sw_x - 1, sw_y - 1, swatch_w + 2, swatch_h + 2), QColor(255, 255, 255, 220))
            painter.fillRect(QRectF(sw_x, sw_y, swatch_w, swatch_h), QColor(pal["base"]))
            painter.fillRect(QRectF(sw_x, sw_y, swatch_w * 0.35, swatch_h), QColor(pal["light"]))
            painter.fillRect(QRectF(sw_x + swatch_w * 0.65, sw_y, swatch_w * 0.35, swatch_h), QColor(pal["dark"]))
            painter.fillRect(QRectF(sw_x, sw_y + swatch_h - 2, swatch_w, 2), QColor(pal["dark"]))

            if is_active_cape:
                pip_r = max(2.0, 2.6 * scale)
                painter.setBrush(QBrush(QColor(BLOSSOM_X6["WHITE"])))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(csx + slot_w - pip_r * 1.5, cape_sy + pip_r * 1.5), pip_r, pip_r)

        div2_y = cape_sy + slot_h_cape + (row_gap / 2.0)
        painter.setPen(QPen(QColor(BLOSSOM_X6["DEEP_BERRY"]), 1.0))
        painter.drawLine(QPointF(target_x + pad_x + 4.0, div2_y), QPointF(target_x + inv_w - pad_x - 4.0, div2_y))

        shadow_sy = cape_sy + slot_h_cape + row_gap
        for k, skey in enumerate(item_keys):
            ssx = target_x + pad_x + k * (slot_w + slot_gap)
            s_rect = QRectF(ssx, shadow_sy, slot_w, slot_h_cape)
            self.inventory_shadow_slot_rects.append((skey, s_rect))

            is_active_shadow = (self.show_shadow and skey == getattr(self, "active_shadow_color", "red"))
            is_hover_shadow = (hover_slot == ("SHADOW", skey))

            if is_active_shadow:
                painter.setBrush(QBrush(QColor(179, 0, 60, 160)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["WHITE"]), max(1.5, 1.8 * scale)))
            elif is_hover_shadow:
                painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
                painter.setPen(QPen(QColor(BLOSSOM_X6["WHITE"]), 1.4))
            else:
                painter.setBrush(QBrush(QColor(18, 18, 24, 210)))
                painter.setPen(QPen(QColor(60, 60, 75, 120), 1.0))

            painter.drawRoundedRect(s_rect, 4.0 * scale, 4.0 * scale)

            pal = CAPE_PALETTES.get(skey, CAPE_PALETTES["red"])
            swatch_w = max(22.0, 26.0 * scale)
            swatch_h = max(10.0, 12.0 * scale)
            sw_x = ssx + (slot_w - swatch_w) / 2.0
            sw_y = shadow_sy + (slot_h_cape - swatch_h) / 2.0
            painter.fillRect(QRectF(sw_x - 1, sw_y - 1, swatch_w + 2, swatch_h + 2), QColor(255, 255, 255, 220))
            painter.fillRect(QRectF(sw_x, sw_y, swatch_w, swatch_h), QColor(pal["dark"]))

            if is_active_shadow:
                pip_r = max(2.0, 2.6 * scale)
                painter.setBrush(QBrush(QColor(BLOSSOM_X6["WHITE"])))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(ssx + slot_w - pip_r * 1.5, shadow_sy + pip_r * 1.5), pip_r, pip_r)

        self.inventory_slot_rects = [r for _, r in self.inventory_hat_slot_rects] + [r for _, r in self.inventory_cape_slot_rects] + [r for _, r in self.inventory_shadow_slot_rects]
        painter.restore()

    def toggle_pointers(self):
        self.show_pointers = not self.show_pointers
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            if self.show_pointers or self.show_inventory:
                self.show_pointers = False
                self.show_inventory = False
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    def changeEvent(self, event):
        if event.type() in (QEvent.Type.ActivationChange, QEvent.Type.FocusOut):
            if not self.isActiveWindow():
                if self.show_pointers or self.show_inventory:
                    self.show_pointers = False
                    self.show_inventory = False
                    self.update()
        super().changeEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.show_pointers = True
            self.update()
            event.accept()

    def mousePressEvent(self, event):
        pos = event.position().toPoint()
        cx = getattr(self, "last_head_cx", self.width() / 2.0)
        cy = getattr(self, "last_head_cy", self.height() / 2.0)

        # 1. Map click coordinates back through the 3D rotation transform for eye clicks
        transform = QTransform()
        transform.translate(cx, cy)
        transform.rotate(self.last_head_rot)
        flip_angle_rad = math.radians(self.rot_x)
        cos_x = math.cos(flip_angle_rad)
        scale_y = math.copysign(max(0.06, abs(cos_x)), cos_x)
        transform.scale(1.0, scale_y)
        transform.translate(-cx, -cy)
        
        try:
            local_pos, _ = transform.inverted()[0].map(QPointF(pos))
            if math.hypot(local_pos.x() - self.last_eye_l.x(), local_pos.y() - self.last_eye_l.y()) <= self.eye_hit_radius:
                self.manual_blink_l_timer = 1.0
                event.accept()
                return
            if math.hypot(local_pos.x() - self.last_eye_r.x(), local_pos.y() - self.last_eye_r.y()) <= self.eye_hit_radius:
                self.manual_blink_r_timer = 1.0
                event.accept()
                return
        except Exception:
            pass

        if self.show_inventory:
            clicked_slot = self.hit_test_inventory(pos)
            if clicked_slot is not None:
                slot_type, slot_val = clicked_slot
                if slot_type == "HAT":
                    if self.show_cap and self.active_hat_idx == slot_val:
                        self.show_cap = False
                    else:
                        self.show_cap = True
                        self.set_active_hat(slot_val)
                elif slot_type == "CAPE":
                    if self.show_cape and self.active_cape_color == slot_val:
                        self.show_cape = False
                    else:
                        self.show_cape = True
                        self.active_cape_color = slot_val
                elif slot_type == "SHADOW":
                    if self.show_shadow and self.active_shadow_color == slot_val:
                        self.show_shadow = False
                    else:
                        self.show_shadow = True
                        self.active_shadow_color = slot_val
                self.save_settings()
                self.update()
                event.accept()
                return
            if self.is_inside_inventory(pos):
                event.accept()
                return

        hit = self.hit_test_handles(pos)
        if hit == "TOP":
            self.scaling_active = True
            self.scale_start_mouse_y = event.globalPosition().y()
            self.scale_start_dim = self.width()
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        elif hit == "RIGHT":
            self.handle_rotating = True
            self.rotate_start_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
            self.rotate_initial_angle = self.manual_rotation_z
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        elif hit == "PENCIL":
            self.show_inventory = not self.show_inventory
            self.update()
            event.accept()
            return

        if self.show_pointers or self.show_inventory:
            self.show_pointers = False
            self.show_inventory = False
            self.update()

        is_rot_modifier = bool(event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier))
        if event.button() == Qt.MouseButton.LeftButton:
            if is_rot_modifier:
                self.rotating = True
                self.rotate_start_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
                self.rotate_initial_angle = self.manual_rotation_z
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            if self.is_in_resize_corner(pos):
                self.resizing = True
                self.resize_start_pos = event.globalPosition().toPoint()
                self.resize_start_size = self.size()
            else:
                # Normal body click: trigger bouncy squish
                self.squash_x = 1.35
                self.squash_y = 0.75
                self.dragging = True
                self.drag_start_global = event.globalPosition().toPoint()
                self.window_start_pos = self.pos()
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.rotating = True
            self.rotate_start_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
            self.rotate_initial_angle = self.manual_rotation_z
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_press_pos = event.globalPosition().toPoint()
            self.rotate_start_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
            self.rotate_initial_angle = self.manual_rotation_z
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        cx = getattr(self, "last_head_cx", self.width() / 2.0)
        cy = getattr(self, "last_head_cy", self.height() / 2.0)

        if self.scaling_active:
            delta_y = event.globalPosition().y() - self.scale_start_mouse_y
            new_dim = int(max(130, min(680, self.scale_start_dim - delta_y * 2.2)))
            if new_dim != self.width():
                old_center = self.geometry().center()
                self.resize(new_dim, new_dim)
                new_rect = self.rect()
                new_rect.moveCenter(old_center)
                self.move(new_rect.topLeft())
                self.update()
                self.update_click_mask()
            event.accept()
            return

        if self.handle_rotating:
            curr_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
            diff = curr_angle - self.rotate_start_angle
            self.manual_rotation_z = (self.rotate_initial_angle + diff) % 360.0
            self.update()
            self.update_click_mask()
            event.accept()
            return

        if self.show_inventory:
            hover_slot = self.hit_test_inventory(pos)
            if hover_slot != getattr(self, "hover_inventory_slot", None):
                self.hover_inventory_slot = hover_slot
                self.hover_inventory_idx = hover_slot
                self.update()
            if hover_slot is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                event.accept()
                return

        hit = self.hit_test_handles(pos)
        old_top_hover = self.hover_top_handle
        old_r_hover = self.hover_right_handle
        old_p_hover = getattr(self, "hover_pencil_handle", False)
        self.hover_top_handle = (hit == "TOP")
        self.hover_right_handle = (hit == "RIGHT")
        self.hover_pencil_handle = (hit == "PENCIL")
        if (old_top_hover != self.hover_top_handle or old_r_hover != self.hover_right_handle or old_p_hover != self.hover_pencil_handle):
            self.update()

        if hit == "TOP":
            self.setCursor(Qt.CursorShape.SizeVerCursor)
            event.accept()
            return
        elif hit in ("RIGHT", "PENCIL"):
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            event.accept()
            return

        if self.rotating:
            curr_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
            diff = curr_angle - self.rotate_start_angle
            self.manual_rotation_z = (self.rotate_initial_angle + diff) % 360.0
            self.update()
            self.update_click_mask()
            event.accept()
            return

        if (event.buttons() & Qt.MouseButton.RightButton) and self.right_press_pos is not None:
            dist = (event.globalPosition().toPoint() - self.right_press_pos).manhattanLength()
            if dist > 5:
                self.rotating = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                curr_angle = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
                diff = curr_angle - self.rotate_start_angle
                self.manual_rotation_z = (self.rotate_initial_angle + diff) % 360.0
                self.update()
                self.update_click_mask()
                event.accept()
                return

        if self.resizing and self.resize_start_size is not None:
            delta = event.globalPosition().toPoint() - self.resize_start_pos
            new_dim = max(130, min(650, self.resize_start_size.width() + delta.x()))
            self.resize(new_dim, new_dim)
            self.update_click_mask()
            event.accept()
        elif self.dragging and (event.buttons() & Qt.MouseButton.LeftButton):
            delta = event.globalPosition().toPoint() - self.drag_start_global
            self.move(self.window_start_pos + delta)
            event.accept()
        else:
            if event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier):
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            elif self.is_in_resize_corner(pos):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self.scaling_active:
            self.scaling_active = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self.handle_rotating:
            self.handle_rotating = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if self.rotating:
            self.rotating = False
            self.right_press_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return

        self.dragging = False
        self.resizing = False
        self.right_press_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        if event.modifiers() & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier):
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta != 0:
                step = 5.0 if delta > 0 else -5.0
                self.manual_rotation_z = (self.manual_rotation_z + step) % 360.0
                self.update()
                self.update_click_mask()
            event.accept()
            return

        delta = event.angleDelta().y()
        if delta == 0: return
        step = 16 if delta > 0 else -16
        new_dim = max(130, min(650, self.width() + step))
        if new_dim != self.width():
            old_dim = self.width()
            self.resize(new_dim, new_dim)
            diff = (new_dim - old_dim) // 2
            self.move(self.x() - diff, self.y() - diff)
            self.update_click_mask()
        event.accept()

    def contextMenuEvent(self, event):
        if self.rotating: return
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: #0b1220;
                border: 1px solid {self.theme['rim_color']};
                border-radius: 8px;
                padding: 4px;
                color: #e0f2fe;
                font-family: -apple-system, sans-serif;
                font-size: 12px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background-color: {self.theme['rim_color']};
                color: #000000;
            }}
            QMenu::separator {{
                height: 1px;
                background: rgba(255, 255, 255, 0.15);
                margin: 4px 8px;
            }}
        """)

        rot_menu = menu.addMenu(f"🔄 Rotate ({int(self.manual_rotation_z)}°)")
        r_right = QAction("Rotate 90° Clockwise (↷)", self)
        r_right.triggered.connect(lambda: self.rotate_by(90))
        rot_menu.addAction(r_right)
        r_left = QAction("Rotate 90° Counter-Clockwise (↶)", self)
        r_left.triggered.connect(lambda: self.rotate_by(-90))
        rot_menu.addAction(r_left)
        r_180 = QAction("Rotate 180° (Upside Down)", self)
        r_180.triggered.connect(lambda: self.rotate_by(180))
        rot_menu.addAction(r_180)
        r_reset = QAction("Reset Rotation (0°)", self)
        r_reset.triggered.connect(self.reset_rotation)
        rot_menu.addAction(r_reset)

        flip_act = QAction("🌀 Perform 360° X-Axis Flip", self)
        flip_act.triggered.connect(self.trigger_360_flip_x)
        menu.addAction(flip_act)

        pointers_toggle = QAction("📍 Scale & Rotate Pointers Visible", self)
        pointers_toggle.setCheckable(True)
        pointers_toggle.setChecked(self.show_pointers)
        pointers_toggle.triggered.connect(self.toggle_pointers)
        menu.addAction(pointers_toggle)

        cam_toggle = QAction("📷 Camera Tracking Enabled", self)
        cam_toggle.setCheckable(True)
        cam_toggle.setChecked(self.tracker.camera_enabled)
        cam_toggle.triggered.connect(self.toggle_camera)
        menu.addAction(cam_toggle)

        menu.addSeparator()

        theme_menu = menu.addMenu("🎨 3D Liquid Theme")
        for key, info in THEMES.items():
            t_act = QAction(info["name"], self)
            t_act.triggered.connect(lambda checked=False, k=key: self.set_theme(k))
            theme_menu.addAction(t_act)

        size_menu = menu.addMenu("📏 Widget Preset Size")
        for label, sz in [("Small (160px)", 160), ("Normal (240px)", 240), ("Large (340px)", 340), ("Giant (460px)", 460)]:
            s_act = QAction(label, self)
            s_act.triggered.connect(lambda checked=False, s=sz: self.set_preset_size(s))
            size_menu.addAction(s_act)

        cap_act = QAction("🧢 Crimson Pixel Cap", self)
        cap_act.setCheckable(True)
        cap_act.setChecked(self.show_cap)
        cap_act.triggered.connect(self.toggle_cap)
        menu.addAction(cap_act)

        ghost_act = QAction("👻 Ghost Mode (100% Click-Through)", self)
        ghost_act.setCheckable(True)
        ghost_act.setChecked(self.click_through_mode)
        ghost_act.triggered.connect(self.toggle_click_through)
        menu.addAction(ghost_act)

        menu.addSeparator()

        quit_act = QAction("❌ Close 3D Water Droplet", self)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)

        menu.exec(event.globalPos())

    def set_preset_size(self, size):
        self.resize(size, size)
        self.update_click_mask()

    def toggle_camera(self):
        self.tracker.camera_enabled = not self.tracker.camera_enabled

    def toggle_cap(self):
        self.show_cap = not self.show_cap
        self.save_settings()
        self.update()

    def set_theme(self, key):
        if key in THEMES:
            self.theme_key = key
            self.theme = THEMES[key]
            self.save_settings()

    def toggle_click_through(self):
        self.click_through_mode = not self.click_through_mode
        self.update_click_mask()

    def closeEvent(self, event):
        if hasattr(self, "timer_72fps"): self.timer_72fps.stop()
        if hasattr(self, "tracker"): self.tracker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    theme = "base"
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            theme = arg.lower()
            break
    app.setApplicationName("Face Puppet Companion (72 FPS)")
    app_icon = get_software_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    puppet = FacePuppetPet(theme)
    puppet.show()
    make_permanent_always_on_top(puppet)
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
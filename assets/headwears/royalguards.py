#!/usr/bin/env python3
"""
ROYAL GUARD BEARSKIN HAT HEADWEAR
=================================
Tall black fur bearskin guardsman ceremonial headwear with chin strap.
"""

import os
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from PySide6.QtCore import Qt

DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_PATH = os.path.join(DIR, "royalguards.png")


class RoyalGuardHeadwear:
    id = "royalguards"
    name = "Royal Guard"
    sprite_filename = "royalguards.png"
    sprite_path = SPRITE_PATH
    base_fit_w = 78.0
    offset_x = 0.0
    offset_y = 28.0
    tuck_y = -0.22

    @classmethod
    def load(cls):
        """Loads sprite, auto-crops visible bounding box, and creates white and ruby/deep-berry silhouettes."""
        path = cls.sprite_path
        if not os.path.exists(path):
            base = getattr(sys, '_MEIPASS', '')
            alt = os.path.join(base, "assets", "headwears", cls.sprite_filename)
            if os.path.exists(alt):
                path = alt
            else:
                return None

        raw_pm = QPixmap(path)
        if raw_pm.isNull():
            return None

        img = raw_pm.toImage()
        min_x, min_y = img.width(), img.height()
        max_x, max_y = -1, -1
        for y in range(img.height()):
            for x in range(img.width()):
                if img.pixelColor(x, y).alpha() > 10:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y

        if max_x >= min_x and max_y >= min_y:
            crop_w = max_x - min_x + 1
            crop_h = max_y - min_y + 1
            cropped_pm = raw_pm.copy(min_x, min_y, crop_w, crop_h)
        else:
            crop_w, crop_h = raw_pm.width(), raw_pm.height()
            cropped_pm = raw_pm

        fit_h = cls.base_fit_w * (crop_h / float(crop_w))

        # 1px crisp white border silhouette
        white_sil = QPixmap(cropped_pm.size())
        white_sil.fill(Qt.GlobalColor.transparent)
        sp = QPainter(white_sil)
        sp.drawPixmap(0, 0, cropped_pm)
        sp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        sp.fillRect(white_sil.rect(), QColor(255, 255, 255, 255))
        sp.end()

        # Deep berry silhouette for 3D perspective shadow facet
        berry_sil = QPixmap(cropped_pm.size())
        berry_sil.fill(Qt.GlobalColor.transparent)
        bp = QPainter(berry_sil)
        bp.drawPixmap(0, 0, cropped_pm)
        bp.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        bp.fillRect(berry_sil.rect(), QColor("#b3003c"))
        bp.end()

        return {
            "id": cls.id,
            "name": cls.name,
            "filename": cls.sprite_filename,
            "pixmap": cropped_pm,
            "white_sil": white_sil,
            "berry_sil": berry_sil,
            "base_fit_w": cls.base_fit_w,
            "base_fit_h": fit_h,
            "offset_x": cls.offset_x,
            "offset_y": cls.offset_y,
            "tuck_y": cls.tuck_y,
        }

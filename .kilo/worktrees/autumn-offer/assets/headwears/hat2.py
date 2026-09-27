#!/usr/bin/env python3
"""
OFFICER PEAKED CAP HEADWEAR
===========================
Military peaked officer cap with gold band emblem.
"""

import os
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from PySide6.QtCore import Qt

DIR = os.path.dirname(os.path.abspath(__file__))
SPRITE_PATH = os.path.join(DIR, "hat2.png")


class OfficerCapHeadwear:
    id = "hat2"
    name = "Officer Cap"
    sprite_filename = "hat2.png"
    sprite_path = SPRITE_PATH
    base_fit_w = 98.0
    offset_x = -4.0
    offset_y = 25.0
    tuck_y = -0.25

    @classmethod
    def load(cls):
        """Loads sprite, auto-crops visible bounding box, and creates white and ruby/deep-berry silhouettes."""
        if not os.path.exists(cls.sprite_path):
            return None

        raw_pm = QPixmap(cls.sprite_path)
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

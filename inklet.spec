# -*- mode: python ; coding: utf-8 -*-
# ==============================================================================
# INKLET - Cross-platform PyInstaller spec (Linux / Windows / macOS) - SLIM
# ==============================================================================
# Driven by env vars so one spec serves every OS/arch:
#   INKLET_ICON       -> path to platform icon (.icns / .ico / .png)  (optional)
#   INKLET_TARGET_ARCH-> macOS only: "arm64" or "x86_64"              (optional)
# Build with:  pyinstaller inklet.spec --noconfirm --clean
# ==============================================================================
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = os.path.abspath(os.getcwd())
APP_NAME = "Inklet"

# --- Data files (only bundle what exists) --------------------------------------
_candidate_data = [
    ("assets", "assets"),
    ("face_landmarker.task", "."),
    ("cap.png", "."),
    ("wizard.png", "."),
    ("base.png", "."),
    ("icon.png", "."),
    ("models", "models"),
    ("face_tracker.py", "."),
    ("cloath_physics.py", "."),
    ("puppet.py", "."),
    ("wizard.py", "."),
]
datas = []
for src, dst in _candidate_data:
    if os.path.exists(os.path.join(ROOT, src)):
        datas.append((src, dst))

binaries = []
# App only touches QtCore/QtGui/QtWidgets — let PyInstaller's PySide6 hook pull
# just those (NOT collect_all, which drags in all of Qt incl. WebEngine/Quick3D).
hiddenimports = [
    "PIL._tkinter_finder",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# --- Heavy packages that genuinely need everything collected -------------------
for pkg in ("mediapipe",):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[inklet.spec] WARN collect_all({pkg}) failed: {e}")

# opencv: rely on PyInstaller's built-in hook (collect_all cv2 triggers a
# recursive-import loop), but still grab its data files just in case.
try:
    datas += collect_data_files("cv2")
except Exception:
    pass
hiddenimports += ["cv2"]

# --- Trim: exclude Qt modules and packages the app never imports ---------------
excludes = [
    "face_puppet_pet",
    # Heavy Qt subsystems (none used):
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtSpatialAudio",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtDesigner", "PySide6.QtUiTools", "PySide6.QtHelp",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.Qt3DAnimation",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtSensors", "PySide6.QtSerialPort",
    "PySide6.QtSerialBus", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtStateMachine",
    "PySide6.QtTextToSpeech", "PySide6.QtHttpServer", "PySide6.QtVirtualKeyboard",
    # Other heavy libs not imported:
    "PyQt5", "PyQt6", "matplotlib", "pandas", "IPython", "jupyter",
    "notebook", "tornado", "pytest",
]

# --- Platform bits -------------------------------------------------------------
icon = os.environ.get("INKLET_ICON") or None
if icon and not os.path.exists(icon):
    icon = None

target_arch = os.environ.get("INKLET_TARGET_ARCH") or None  # macOS only

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # windowed / GUI app
    disable_windowed_traceback=False,
    argv_emulation=(sys.platform == "darwin"),
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# --- macOS .app bundle ---------------------------------------------------------
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier="com.shoroz.inklet",
        info_plist={
            "NSCameraUsageDescription":
                "Inklet requires camera access for real-time facial "
                "tracking and expression animation.",
            "NSHighResolutionCapable": True,
            "LSUIElement": False,
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
        },
    )

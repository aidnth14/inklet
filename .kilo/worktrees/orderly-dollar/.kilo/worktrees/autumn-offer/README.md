# Face Puppet 🎭

> **High-Precision 1:1 Facial Desktop Pet & Kinetic Puppet**  
> An interactive, always-on-top retro pixel art desktop companion with real-time 3D head pose tracking, decoupled cervical neck kinematics, independent gaze and blink detection, Verlet cloth physics, and a modular headwear inventory.

---

## 🌟 Highlights & Features

### 1. 🎯 Precision 3D Facial Kinematics
- **3D Head Pose Decomposition**: Solves true 3D Euler angles (Yaw, Pitch, Roll) and relative camera depth ($Z$) using facial transformation matrix analysis.
- **Decoupled Cervical Spine**: Independent neck anchor kinematics ($152 \to \text{cervical pivot}$) providing organic neck bending, nodding, and kinetic drag.
- **Strict Resting Deadbands**: Adaptive baseline calibration gently learns resting desk posture, completely eliminating micro-twitches and floating jitters when sitting still.

### 2. 👁️ Independent Eye Gaze & Real Blink Detection
- **Laser Gaze Vector & Compass HUD**: High-precision pupillary corneal reflection (PCCR) tracking calculating gaze horizontal and vertical vectors with real-time 360° circular compass telemetry.
- **Independent Blinks & Winks**: Detects individual left and right eyelid closures for natural single-eye winks and synchronized blinks.
- **Eyebrow-to-Eye Dynamics**: Measures eyebrow arch elevation and furrowing to modulate character eye aperture (surprised wide-eyes vs. concentrated squints) while maintaining purely retro two-dot pixel eyes.

### 3. 👄 Speech Lip-Sync & Viseme Engine
- **Independent Mouth Kinematics**: Traces lateral displacement, oral aperture height, and horizontal mouth span relative to facial features.
- **Viseme Classification**: Dynamically identifies speech syllables and emotional expressions (`WIDE_GRIN`, `SMIRK`, `TRIANGLE_SMILE`, `O_PHONEME`, `OPEN_JAW`, `FROWN`, and `NEUTRAL`).
- **Teeth Exposure**: Tracks separate upper and lower incisor exposure during speech and smiles.

### 4. 👓 Glasses Anti-Jitter & Optical Auto-Tuner
- **Specular Glare Rejection**: Cross-validates geometric Eyelid Aspect Ratio (EAR) against blendshapes to discard false blink spikes from eyeglass reflections and frame shadows.
- **Closed-Loop Optical Tuner**: Auto-tunes non-linear gamma lifts, adaptive CLAHE contrast, and specular roll-off in real-time until tracking accuracy reaches 100%.

### 5. 🧥 Kinetic Cloth & Cape Mechanics
- **Verlet Wave Integration**: Physics-driven crimson cape with natural cloth draping, dynamic gravity sag, and fluid ripples responding to head turns and window drags.
- **Pixelated Retro Bordering**: Rendered using strictly pixelated contour algorithms with zero smooth vector lines.

### 6. 🎩 Modular Headwear & Inventory System
- **Floating Cap Physics**: Hats operate as distinct physics instances with momentum, inertia lag, and damped spring-return when dragged fast.
- **Drop-Down Wardrobe Inventory**: Interactive pencil icon marker displays a wardrobe selector directly underneath the cape.
- **Modular Plugin Architecture**: Each headwear is completely self-contained in its own module inside `assets/headwears/` with tailored sizing, brim offsets, and cranium tuck thresholds:
  - **Crimson Beret** (`cap.py`)
  - **Officer Peaked Cap** (`hat2.py`)
  - **Royal Guard Bearskin** (`royalguards.py`)
  - **Holiday Santa Hat** (`christmashat.py`)

---

## 📁 Repository Structure

```text
face-puppet/
├── assets/
│   └── headwears/               # Modular headwear plugins & sprites
│       ├── cap.py               # Crimson Beret configuration
│       ├── hat2.py              # Officer Peaked Cap configuration
│       ├── royalguards.py       # Royal Guard Bearskin configuration
│       ├── christmashat.py      # Holiday Santa Hat configuration
│       └── *.png                # Headwear pixel art sprites
├── face_puppet_pet.py           # 1. Main desktop pet application & rendering
├── face_tracker.py              # 2. High-performance background tracking thread
├── cloath_physics.py            # 3. Verlet integration cloth physics simulation
├── run.sh                       # 4. Quick launcher script
├── README.md                    # 5. Project documentation
├── LICENSE.md                   # 6. MIT Open Source License
└── .gitignore                   # 7. Comprehensive build & checkpoint exclusions
```

---

## 🚀 Getting Started

### Prerequisites
- **Python**: 3.10 – 3.14
- **Operating System**: macOS (Apple Silicon & Intel), Windows 10/11, or Linux

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/face-puppet.git
   cd face-puppet
   ```

2. **Install dependencies**:
   ```bash
   pip install PySide6 opencv-python mediapipe numpy scipy torch torchvision onnx onnxscript
   ```

3. **Launch the application**:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   *Or launch directly via Python:*
   ```bash
   python3 face_puppet_pet.py
   ```

---

## 🎮 Controls & Interactions

| Action | Control | Effect |
| :--- | :--- | :--- |
| **Move Pet** | `Left-Click + Drag` | Drags character across screen; hat lags behind with inertial physics. |
| **Flick / Shake** | `Rapid Mouse Drag` | Triggers inertial hat fall-off and smooth elastic recovery. |
| **Wardrobe Inventory** | `Click Pencil Marker` | Opens headwear wardrobe underneath cape. Click anywhere else to close. |
| **Toggle Headwear** | `Double-Click Hat` | Hides or shows current headwear. |
| **Context Menu** | `Right-Click` | Opens menu to switch themes, toggle live audit HUD, or mute camera. |
| **360° Flip / Spin** | `Kinetic Head Jerk` | Character executes a celebratory 360-degree mid-air flip. |

---

## 🧠 Model Training & Dataset Pipeline

Face Puppet includes a dedicated training pipeline in `training/` designed to train on **only targeted facial features**:
- **Eyes & Pupils**: 3.0x loss weight for sub-pixel gaze and blink accuracy.
- **Eyebrows**: 2.5x loss weight for brow elevation and eye aperture scaling.
- **Lips & Mouth**: 2.5x loss weight for speech phonemes and smile detection.
- **Neck Anchor**: 2.2x loss weight for decoupled cervical spine connection.
- **Face / Head**: 1.5x loss weight for 3D Euler angles (Yaw, Pitch, Roll).
- **Excluded**: Hair, beard, clothing, glasses frames, and background are strictly masked out.

### Supported Datasets:
1. **WFLW** (`PFLD-Pytorch-Landmarks-98-master`): 98-point landmark annotation with 3D Euler angles.
2. **Face Synthetics** (`FaceSynthetics-main`): 70 landmarks and semantic segmentation masks.
3. **LaPa** (`lapa-dataset-master`): 106 landmarks and boundary-guided facial parsing.

### Training Command:
```bash
python3 training/train.py \
    --epochs 25 \
    --batch_size 16 \
    --lr 0.0001 \
    --device auto \
    --save_dir models
```
*`--device auto` automatically activates Apple Silicon GPU Metal (`mps`) or NVIDIA GPU (`cuda`).*

### Export to ONNX:
```bash
python3 training/export_onnx.py \
    --checkpoint models/face_puppet_features_best.pth \
    --output models/face_puppet_features.onnx
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

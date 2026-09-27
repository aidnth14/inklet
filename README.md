<div align="center">
  <img src="assets/publish/inklet.gif" alt="Face Puppet in Action" width="240" style="border-radius: 12px; margin-bottom: 20px;">
  
  <h1>Face Puppet</h1>
  
  <p><b>High-Precision 1:1 Facial Desktop Pet & Kinetic Puppet</b></p>
</div>

<p align="center">
  An interactive, always-on-top retro pixel art desktop companion featuring absolute 3D facial tracking, hybrid 3D Euclidean & AI blendshape blink/mouth kinematics, zero baseline drift, Verlet cloth physics, and a modular headwear inventory.
</p>

<div align="center">
  <img src="assets/publish/Screenshot 2026-09-26 at 10.00.49 PM.jpg" alt="Desktop Integration" width="45%" style="border-radius: 8px; margin-right: 2%;">
  <img src="assets/publish/Untitled design.png" alt="Expression Range" width="45%" style="border-radius: 8px;">
</div>

<br>

## Highlights & Features

### 1. Absolute 3D Facial Kinematics
- **3D Head Pose Decomposition**: Solves true 3D Euler angles (Yaw, Pitch, Roll) and relative camera depth (Z) using facial transformation matrix analysis.
- **Decoupled Cervical Spine**: Independent neck anchor kinematics providing organic neck bending, nodding, and kinetic drag.
- **Zero Baseline Drift**: "Forward" is locked permanently. No need to look rigidly straight into the camera for calibration.
- **Dynamic Z-Depth Scaling**: The character physically resizes dynamically in real-time based on how close or far your face is from the camera.

### 2. Hybrid 3D Geometry & Blendshape Blinks
- **Guaranteed Two-Eye Blinks**: Combines 3D Euclidean Eyelid Aspect Ratio (EAR) calculations with AI blendshapes, taking the maximum of both to completely eliminate the "one-eye blink" detection glitch.
- **Hyper-Sensitive Lip-Sync**: Dual-sensor mouth tracking combines physical 3D lip separation with jaw open blendshapes so subtle murmurs trigger fluid mouth movements instantly without exaggeration.
- **Interactive Click-to-Blink**: Clicking directly on either eye triggers a smooth, independent manual blink animation.
- **Eyebrow-to-Eye Dynamics**: Modulates character eye aperture dynamically based on eyebrow movements.

### 3. Speech Lip-Sync & Viseme Engine
- **Independent Mouth Kinematics**: Traces lateral displacement, oral aperture height, and horizontal mouth span relative to facial features.
- **Viseme Classification**: Dynamically identifies speech syllables and emotional expressions (`WIDE_GRIN`, `SMIRK`, `TRIANGLE_SMILE`, `ROUNDED_O`, `PUCKER_U`, `OPEN_JAW`, `FROWN`, and `NEUTRAL`).
- **Teeth Exposure**: Tracks separate upper and lower incisor exposure during speech and smiles.

### 4. Zero-Latency Asynchronous Pipeline
- **FastVideoCapture Daemon**: Grabs webcam frames on a dedicated background thread, bypassing USB bottlenecks to achieve true millisecond-level responsiveness.
- **Closed-Loop Optical Tuner**: Auto-tunes gamma lifts, adaptive CLAHE contrast, and specular roll-off in real-time to maintain 100% tracking accuracy across changing lighting conditions.

### 5. Kinetic Cloth & Cape Mechanics
*Drawing inspiration from the live avatar work of DaFluffyPotato, this subsystem offers a physics-based pixel-art cape engine specifically tailored for avatar tracking.*

- **Verlet Integration**: Performs a Provot-style sub-stepping scheme three times per frame to ensure numerical stability during rapid head turns and flipping events.
- **Structural & Shear Constraints**: Enforces uniform grid spacing and prevents fabric deformation under rotational torque.
- **Procedural Wind**: Rolling motion energy estimator assesses avatar activity to dynamically estimate wind turbulence.

### 6. Interactive Physics & Particles
- **Transparent Click Mask**: Mouse clicks pass naturally through transparent pixels. Only the visible pixel-art shape of the pet is clickable, leaving your desktop fully accessible.
- **Bouncy Body Squish**: Clicking anywhere on the body instantly squishes the character with an elastic bounce recovery.
- **Liquid Drag Particles**: A dynamic black liquid spray particle system triggers when you drag the pet quickly across the screen.

### 7. Modular Headwear & Inventory System
- **Floating Cap Physics**: Hats operate as distinct physics instances with momentum, inertia lag, and magnetic spring-return levitation when dragged fast.
- **Drop-Down Wardrobe Inventory**: Interactive pencil icon marker displays a wardrobe selector directly underneath the cape.
- **Persistent Saves**: Remembers your chosen hat, cape color, shadow color, and 3D theme across sessions using `QSettings`.

---

## Architecture & Rendering Pipeline

### Physics Engine Design
All physics computation occurs in a collar-anchored local coordinate system. Avatar inputs are translated into physical forces that push and pull cape nodes based on drag velocity, head yaw, pitch, and tilt.

### Rendering Pipeline
- **Gapless Pixel-Art Upscaling**: Rasterizes geometry onto a low-resolution offscreen ARGB buffer with an overlap extension factor, completely eliminating black hairline grid artifacts between pixels. Upscaling via nearest-neighbor sampling (`SmoothPixmapTransform = False`) maintains crisp pixel edges.
- **Dynamic Outlines**: Blits multi-directional offset mask layers to form a dilated white silhouette.
- **Directional Palette Shading**: Evaluates local quad shear strain to dynamically transition colors across base, highlight, and shadow palettes.

---

## Getting Started

### Prerequisites
- **Python**: 3.10 – 3.14
- **Operating System**: macOS (Apple Silicon & Intel), Windows 10/11, or Linux

### Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/face-puppet.git](https://github.com/your-username/face-puppet.git)
   cd face-puppet
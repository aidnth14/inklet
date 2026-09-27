<div align="center">
  <img src="assets/publish/inklet.gif" alt="Face Puppet in Action" width="240" style="border-radius: 12px; margin-bottom: 20px;">
  
  <h1>Face Puppet</h1>
  
  <p><b>High-Precision 1:1 Facial Desktop Pet & Kinetic Puppet</b></p>
</div>

<p align="center">
  An interactive, always-on-top retro pixel art desktop companion with real-time 3D head pose tracking, decoupled cervical neck kinematics, independent gaze and blink detection, Verlet cloth physics, and a modular headwear inventory.
</p>

<div align="center">
  <img src="assets/publish/Screenshot 2026-09-26 at 10.00.49 PM.jpg" alt="Desktop Integration" width="45%" style="border-radius: 8px; margin-right: 2%;">
  <img src="assets/publish/Untitled design.png" alt="Expression Range" width="45%" style="border-radius: 8px;">
</div>

<br>

## Highlights & Features

### 1. Precision 3D Facial Kinematics
- **3D Head Pose Decomposition**: Solves true 3D Euler angles (Yaw, Pitch, Roll) and relative camera depth (Z) using facial transformation matrix analysis.
- **Decoupled Cervical Spine**: Independent neck anchor kinematics (152 -> cervical pivot) providing organic neck bending, nodding, and kinetic drag.
- **Strict Resting Deadbands**: Adaptive baseline calibration gently learns resting desk posture, completely eliminating micro-twitches and floating jitters when sitting still.
- **Dynamic Z-Depth Scaling**: The character physically resizes dynamically in real-time based on how close or far your face is from the camera.

### 2. Independent Eye Gaze & Real Blink Detection
- **Laser Gaze Vector & Compass HUD**: High-precision pupillary corneal reflection (PCCR) tracking calculating gaze horizontal and vertical vectors with real-time 360° circular compass telemetry.
- **Independent Blinks & Winks**: Detects individual left and right eyelid closures for natural single-eye winks and synchronized blinks.
- **Interactive Click-to-Blink**: Clicking directly on either eye triggers a smooth, independent manual blink animation.
- **Eyebrow-to-Eye Dynamics**: Measures eyebrow arch elevation and furrowing to modulate character eye aperture (surprised wide-eyes vs. concentrated squints) while maintaining purely retro two-dot pixel eyes.

### 3. Speech Lip-Sync & Viseme Engine
- **Independent Mouth Kinematics**: Traces lateral displacement, oral aperture height, and horizontal mouth span relative to facial features.
- **Viseme Classification**: Dynamically identifies speech syllables and emotional expressions (`WIDE_GRIN`, `SMIRK`, `TRIANGLE_SMILE`, `O_PHONEME`, `OPEN_JAW`, `FROWN`, and `NEUTRAL`).
- **Teeth Exposure**: Tracks separate upper and lower incisor exposure during speech and smiles.

### 4. Glasses Anti-Jitter & Optical Auto-Tuner
- **Specular Glare Rejection**: Cross-validates geometric Eyelid Aspect Ratio (EAR) against blendshapes to discard false blink spikes from eyeglass reflections and frame shadows.
- **Closed-Loop Optical Tuner**: Auto-tunes non-linear gamma lifts, adaptive CLAHE contrast, and specular roll-off in real-time until tracking accuracy reaches 100%.

### 5. Kinetic Cloth & Cape Mechanics
*Drawing inspiration from the live avatar work of DaFluffyPotato, this subsystem offers a physics-based pixel-art cape engine specifically tailored for avatar tracking.*

- **Verlet Integration**: Performs a Provot-style sub-stepping scheme three times per frame to ensure numerical stability during rapid head turns, tracking jitter, and flipping events.
- **Structural Constraints**: Enforces uniform grid spacing of the fabric mesh.
- **Diagonal Shear Constraints**: Prevents parallelogram collapse in response to rotational torque.
- **Soft Bend Constraints**: Applies soft bend constraints across skip-one neighbors to allow realistic fabric draping without rigidity or folding artifacts.
- **Procedural Wind**: A rolling motion energy estimator assesses avatar activity to estimate wind turbulence that keeps the cape still when the avatar is inactive and active when the avatar is moving.

### 6. Interactive Physics & Particles
- **Bouncy Body Squish**: Clicking anywhere on the body instantly squishes the character with an elastic bounce recovery.
- **Liquid Drag Particles**: A dynamic black liquid spray particle system triggers when you drag the pet quickly across the screen.

### 7. Modular Headwear & Inventory System
- **Floating Cap Physics**: Hats operate as distinct physics instances with momentum, inertia lag, and magnetic spring-return levitation when dragged fast.
- **Drop-Down Wardrobe Inventory**: Interactive pencil icon marker displays a wardrobe selector directly underneath the cape.
- **Persistent Saves**: Remembers your chosen hat, cape color, shadow color, and 3D theme across sessions.

---

## Architecture & Rendering Pipeline

### Physics Engine Design
All physics computation occurs in a collar-anchored local coordinate system, ensuring high reactivity. Avatar inputs are translated into physical forces that push and pull cape nodes based on drag velocity, head yaw, pitch, and tilt. User event triggers (such as facial smiles and flips) provide impulse velocity changes in historical positions of the cape nodes.

### Rendering Pipeline
- **Pixel-Art Upscaling**: Rasterizes cape geometry onto a low-resolution offscreen ARGB buffer and upscales via nearest-neighbor sampling (`SmoothPixmapTransform = False`) to align crisp pixels with character body sprites. Gapless overlap factors ensure solid blocks of color.
- **Dynamic Outlines**: Blits multi-directional offset mask layers to form a dilated white silhouette.
- **Directional Palette Shading**: Evaluates local quad shear strain to dynamically transition colors across base, highlight, and shadow palettes.
- **Visual Impulses**: Emits a brief warm glow flash across the mesh during physical impulses.

### Dataset & Data Pipeline
Because the simulation relies on deterministic Verlet physics rather than a trained machine learning model, it does not use a traditional offline training dataset. Instead, it operates on a continuous real-time telemetry stream of frame-by-frame tracking variables, including head transformation matrices, positional velocity vectors, and blendshape expression coefficients.

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
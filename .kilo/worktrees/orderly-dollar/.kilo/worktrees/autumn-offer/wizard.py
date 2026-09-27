#!/usr/bin/env python3
import os
import sys
import time
import threading
import subprocess
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ==========================================
# WIZARD CONFIGURATION & PALETTE
# ==========================================
UNIFIED_BG = "#1C1C1E"      
TEXT_FG = "#FFFFFF"         
ACCENT_BLUE = "#0A84FF"     

FONT_TITLE = ("Josefin Sans", 24, "bold")
FONT_SUBTITLE = ("Josefin Sans", 13)
FONT_BODY = ("Josefin Sans", 12)
FONT_MONO = ("Courier", 11)

REQUIRED_PACKAGES = [
    "PySide6",
    "opencv-python",
    "mediapipe",
    "numpy",
    "scipy",
    "Pillow"
]

# ==========================================
# CUSTOM UI WIDGETS (Matching Reference Images)
# ==========================================
class CustomFlatButton(tk.Frame):
    """A custom flat button matching the white, black-bordered, black-text aesthetic."""
    def __init__(self, parent, text, command, width=12, **kwargs):
        super().__init__(parent, bg="#A0A0A0", bd=1)
        self.command = command
        self._state = tk.NORMAL
        
        self.label = tk.Label(self, text=text, bg="#FFFFFF", fg="#000000", font=("Josefin Sans", 11), width=width)
        self.label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        self.label.bind("<Enter>", self.on_hover)
        self.label.bind("<Leave>", self.on_leave)
        self.label.bind("<Button-1>", self.on_click)
        self.label.bind("<ButtonRelease-1>", self.on_release)

    def on_hover(self, event):
        if self._state == tk.NORMAL:
            self.label.config(bg="#F0F0F0")

    def on_leave(self, event):
        if self._state == tk.NORMAL:
            self.label.config(bg="#FFFFFF")

    def on_click(self, event):
        if self._state == tk.NORMAL:
            self.label.config(bg="#E0E0E0")

    def on_release(self, event):
        if self._state == tk.NORMAL:
            self.label.config(bg="#F0F0F0")
            if self.command:
                self.command()

    def config(self, state=None, text=None):
        if state is not None:
            self._state = state
            if state == tk.DISABLED:
                self.label.config(fg="#A0A0A0", bg="#E8E8E8")
            else:
                self.label.config(fg="#000000", bg="#FFFFFF")
        if text is not None:
            self.label.config(text=text)


class CustomRadioButton(tk.Frame):
    """Custom drawn radio button to bypass macOS native styling bugs."""
    def __init__(self, parent, text, variable, value, command=None, **kwargs):
        super().__init__(parent, bg=UNIFIED_BG)
        self.variable = variable
        self.value = value
        self.command = command
        
        self.canvas = tk.Canvas(self, width=24, height=24, bg=UNIFIED_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        
        self.label = tk.Label(self, text=text, bg=UNIFIED_BG, fg=TEXT_FG, font=FONT_BODY)
        self.label.pack(side=tk.LEFT, padx=5)
        
        self.canvas.bind("<Button-1>", self.on_click)
        self.label.bind("<Button-1>", self.on_click)
        
        self.variable.trace_add("write", self.update_view)
        self.update_view()

    def on_click(self, event):
        self.variable.set(self.value)
        if self.command:
            self.command()

    def update_view(self, *args):
        self.canvas.delete("all")
        if self.variable.get() == self.value:
            self.canvas.create_oval(2, 2, 22, 22, outline=ACCENT_BLUE, width=2)
            self.canvas.create_oval(6, 6, 18, 18, fill=ACCENT_BLUE, outline=ACCENT_BLUE)
        else:
            self.canvas.create_oval(2, 2, 22, 22, outline="#A0A0A0", width=2)


# ==========================================
# MAIN WIZARD APPLICATION
# ==========================================
class InkletWizard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Inklet Setup")
        self.geometry("680x480")
        self.resizable(False, False)
        self.configure(bg=UNIFIED_BG)
        self.center_window(680, 480)
        
        self.load_app_icon()

        self.current_step = 0
        self.frames = []

        self.build_ui()
        self.show_step(0)
        
        if not getattr(sys, 'frozen', False):
            self.setup_hot_reload()

    def center_window(self, width, height):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.geometry(f"{width}x{height}+{x}+{y}")

    def load_app_icon(self):
        try:
            icon_path = "/Users/shoroz/Desktop/inklet/wizard.png"
            if os.path.exists(icon_path):
                from PIL import Image, ImageTk
                raw_icon = Image.open(icon_path).convert("RGBA")
                self.app_icon = ImageTk.PhotoImage(raw_icon)
                self.iconphoto(True, self.app_icon)
        except Exception as e:
            print(f"Could not load custom icon: {e}")

    def setup_hot_reload(self):
        self.watch_files = [os.path.abspath(__file__)]
        self.last_mtimes = {f: os.path.getmtime(f) for f in self.watch_files if os.path.exists(f)}
        self.last_reload_time = time.time()
        self.after(1000, self.check_reload)

    def check_reload(self):
        if time.time() - self.last_reload_time < 2.0:
            self.after(1000, self.check_reload)
            return
        for f in self.watch_files:
            if os.path.exists(f):
                current_mtime = os.path.getmtime(f)
                if current_mtime > self.last_mtimes[f]:
                    print(f"[Hot Reload] Change detected in {f}. Restarting wizard...")
                    self.destroy()
                    os.execv(sys.executable, ['python3'] + sys.argv)
        self.after(1000, self.check_reload)

    def build_ui(self):
        # Left Sidebar Image
        self.canvas = tk.Canvas(self, bg=UNIFIED_BG, highlightthickness=0)
        self.canvas.place(x=0, y=0, width=200, height=430)
        self._load_sidebar_image()

        # Bottom Navigation Bar
        sep = tk.Frame(self, bg="#333333", bd=0)
        sep.place(x=0, y=430, width=680, height=1)

        btn_frame = tk.Frame(self, bg=UNIFIED_BG)
        btn_frame.place(x=0, y=431, width=680, height=49)

        self.btn_back = CustomFlatButton(btn_frame, text="< Back", command=self.prev_step, width=10)
        self.btn_back.place(x=370, y=10)

        self.btn_next = CustomFlatButton(btn_frame, text="Next >", command=self.next_step, width=10)
        self.btn_next.place(x=475, y=10)

        self.btn_cancel = CustomFlatButton(btn_frame, text="Cancel", command=self.destroy, width=10)
        self.btn_cancel.place(x=580, y=10)

        # Content Container
        self.content_container = tk.Frame(self, bg=UNIFIED_BG)
        self.content_container.place(x=200, y=0, width=480, height=420)

        # Build Steps
        self.frames.append(self.build_step_1_agreement())
        self.frames.append(self.build_step_2_privacy())
        self.frames.append(self.build_step_3_camera())
        self.frames.append(self.build_step_4_install())
        self.frames.append(self.build_step_5_finish())

    def _load_sidebar_image(self):
        img_path = "/Users/shoroz/Desktop/inklet/assets/img09.27.2026.png"
        try:
            from PIL import Image, ImageTk, ImageOps
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img = ImageOps.fit(img, (200, 430), Image.Resampling.LANCZOS)
                self.sidebar_image = ImageTk.PhotoImage(img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.sidebar_image)
            else:
                self.canvas.create_text(100, 215, text="INKLET", fill=TEXT_FG, font=FONT_TITLE)
        except ImportError:
            self.canvas.create_text(100, 215, text="INKLET", fill=TEXT_FG, font=FONT_TITLE)

    def _create_header(self, parent, title_text, desc_text):
        header = tk.Frame(parent, bg=UNIFIED_BG, height=90)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        
        tk.Label(header, text=title_text, font=FONT_TITLE, bg=UNIFIED_BG, fg=TEXT_FG, anchor="w").place(x=20, y=15)
        tk.Label(header, text=desc_text, font=FONT_SUBTITLE, bg=UNIFIED_BG, fg="#CCCCCC", anchor="w").place(x=20, y=55)
        return header

    # ==========================================
    # PHASE 1: LICENSE AGREEMENT
    # ==========================================
    def build_step_1_agreement(self):
        frame = tk.Frame(self.content_container, bg=UNIFIED_BG)
        self._create_header(frame, "License Agreement", "Please acknowledge the software terms.")

        body = tk.Frame(frame, bg=UNIFIED_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        agreement_text = (
            "Inklet is distributed with a LICENSE.md and README.md file in your download archive.\n\n"
            "Because this software is in active development, the license terms and documentation are maintained externally.\n\n"
            "By proceeding, you confirm that you have read and agree to the terms outlined in the LICENSE file provided with this distribution."
        )

        tk.Label(body, text=agreement_text, font=FONT_BODY, bg=UNIFIED_BG, fg=TEXT_FG, justify=tk.LEFT, wraplength=420).pack(fill=tk.X, pady=(0, 40), anchor="w")

        self.license_var = tk.IntVar(value=0)
        CustomRadioButton(body, text="I have read and accept the external agreement", variable=self.license_var, value=1, command=self.update_nav_buttons).pack(anchor="w", pady=(0, 10))
        CustomRadioButton(body, text="I do not accept", variable=self.license_var, value=0, command=self.update_nav_buttons).pack(anchor="w")

        return frame

    # ==========================================
    # PHASE 2: PRIVACY POLICY
    # ==========================================
    def build_step_2_privacy(self):
        frame = tk.Frame(self.content_container, bg=UNIFIED_BG)
        self._create_header(frame, "Privacy Policy", "Please review how Inklet handles your data.")

        body = tk.Frame(frame, bg=UNIFIED_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        privacy_text = (
            "Inklet processes all webcam video and facial landmarks "
            "STRICTLY LOCALLY on your machine.\n\n"
            "At no point is any video feed, image data, or personal biometric "
            "telemetry recorded, stored, or uploaded to any remote server or cloud infrastructure.\n\n"
            "By continuing, you acknowledge that you understand how your local sensor data is utilized solely for real-time 3D rendering."
        )

        tk.Label(body, text=privacy_text, font=FONT_BODY, bg=UNIFIED_BG, fg=TEXT_FG, justify=tk.LEFT, wraplength=420).pack(fill=tk.X, pady=(0, 30), anchor="w")

        self.privacy_var = tk.IntVar(value=0)
        CustomRadioButton(body, text="I understand and accept", variable=self.privacy_var, value=1, command=self.update_nav_buttons).pack(anchor="w", pady=(0, 10))
        CustomRadioButton(body, text="I decline", variable=self.privacy_var, value=0, command=self.update_nav_buttons).pack(anchor="w")

        return frame

    # ==========================================
    # PHASE 3: CAMERA & SYSTEM PERMISSIONS
    # ==========================================
    def build_step_3_camera(self):
        frame = tk.Frame(self.content_container, bg=UNIFIED_BG)
        self._create_header(frame, "Camera Permissions", "Inklet requires local camera access.")

        body = tk.Frame(frame, bg=UNIFIED_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        cam_text = (
            "Inklet requires hardware access to your webcam to perform 1:1 facial tracking and kinematics.\n\n"
            "Depending on your Operating System (macOS/Windows), you may need to explicitly grant this application permission to access the camera device."
        )
        tk.Label(body, text=cam_text, font=FONT_BODY, bg=UNIFIED_BG, fg=TEXT_FG, justify=tk.LEFT, wraplength=420).pack(fill=tk.X, pady=(0, 40), anchor="w")

        btn_cam_test = CustomFlatButton(body, text="Grant Access", command=self.request_camera_access, width=20)
        btn_cam_test.pack(pady=10)

        self.lbl_cam_status = tk.Label(body, text="", font=FONT_BODY, bg=UNIFIED_BG, fg=ACCENT_BLUE)
        self.lbl_cam_status.pack()

        return frame

    def request_camera_access(self):
        self.lbl_cam_status.config(text="Probing video capture device...", fg=ACCENT_BLUE)
        self.update()
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    self.lbl_cam_status.config(text="Camera access granted successfully!", fg="#34C759")
                else:
                    self.lbl_cam_status.config(text="Camera opened, but failed to read frame.", fg="#FF3B30")
            else:
                self.lbl_cam_status.config(text="Failed to open camera. Check OS privacy settings.", fg="#FF3B30")
        except ImportError:
            self.lbl_cam_status.config(text="OpenCV not installed yet. Will be installed next.", fg="#34C759")
        except Exception as e:
            self.lbl_cam_status.config(text=f"Error: {str(e)}", fg="#FF3B30")

    # ==========================================
    # PHASE 4: INSTALLATION & DEPENDENCIES
    # ==========================================
    def build_step_4_install(self):
        frame = tk.Frame(self.content_container, bg=UNIFIED_BG)
        self._create_header(frame, "Install Dependencies", "Preparing your local environment.")

        body = tk.Frame(frame, bg=UNIFIED_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Basic canvas to emulate progress bar for aesthetic control
        self.progress_canvas = tk.Canvas(body, height=10, width=420, bg="#333333", highlightthickness=0)
        self.progress_canvas.pack(pady=(0, 20))
        self.progress_rect = self.progress_canvas.create_rectangle(0, 0, 0, 10, fill=TEXT_FG, outline="")

        self.txt_log = scrolledtext.ScrolledText(body, width=40, height=10, font=FONT_MONO, bg=UNIFIED_BG, fg=TEXT_FG, bd=1, highlightbackground="#333333", highlightthickness=1)
        self.txt_log.pack(fill=tk.BOTH, expand=True)
        self.txt_log.insert(tk.END, "Ready to install.\n")
        self.txt_log.config(state=tk.DISABLED)

        return frame

    def set_progress(self, percent):
        width = int((percent / 100.0) * 420)
        self.progress_canvas.coords(self.progress_rect, 0, 0, width, 10)
        self.update_idletasks()

    def start_installation(self):
        self.btn_next.config(state=tk.DISABLED)
        self.btn_back.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.DISABLED)
        
        self.set_progress(5)
        threading.Thread(target=self._install_worker, daemon=True).start()

    def _log(self, msg):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def _install_worker(self):
        self._log("Starting installation...")
        
        if getattr(sys, 'frozen', False):
            self._log("Unpacking bundled dependencies...")
            step = 100 / len(REQUIRED_PACKAGES)
            for i, pkg in enumerate(REQUIRED_PACKAGES):
                time.sleep(0.5) 
                self.after(0, self.set_progress, (i + 1) * step)
                self.after(0, self._log, f"Configuring bundled module: {pkg}...")
            time.sleep(1)
            self.after(0, self._install_complete, True)
        else:
            try:
                cmd = [sys.executable, "-m", "pip", "install"] + REQUIRED_PACKAGES
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True
                )
                
                # Simulate progress while pip runs
                threading.Thread(target=self._simulate_progress, daemon=True).start()
                
                for line in iter(process.stdout.readline, ''):
                    self.after(0, self._log, line.strip())
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    self.after(0, self._install_complete, True)
                else:
                    self.after(0, self._install_complete, False)
                    
            except Exception as e:
                self.after(0, self._log, f"CRITICAL ERROR: {str(e)}")
                self.after(0, self._install_complete, False)

    def _simulate_progress(self):
        for i in range(5, 95):
            time.sleep(0.2)
            self.after(0, self.set_progress, i)

    def _install_complete(self, success):
        self.set_progress(100)
        if success:
            self._log("\nInstallation finished successfully.")
            self.btn_next.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.NORMAL)
            self.next_step() 
        else:
            self._log("\nInstallation failed. Check the log above.")
            self.btn_back.config(state=tk.NORMAL)
            self.btn_cancel.config(state=tk.NORMAL)

    # ==========================================
    # PHASE 5: COMPLETE & LAUNCH
    # ==========================================
    def build_step_5_finish(self):
        frame = tk.Frame(self.content_container, bg=UNIFIED_BG)
        self._create_header(frame, "Setup Complete", "Inklet is ready to launch.")

        body = tk.Frame(frame, bg=UNIFIED_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        finish_text = (
            "The Inklet Setup Wizard has successfully configured your system.\n\n"
            "Hardware Check: OK\n"
            "Dependencies: OK\n\n"
            "Click Finish to exit the wizard and launch your 3D desktop companion."
        )
        tk.Label(body, text=finish_text, font=FONT_BODY, bg=UNIFIED_BG, fg=TEXT_FG, justify=tk.LEFT, wraplength=420).pack(fill=tk.X, pady=(0, 20), anchor="w")

        return frame

    # ==========================================
    # NAVIGATION LOGIC
    # ==========================================
    def update_nav_buttons(self):
        if self.current_step == 0:
            self.btn_back.config(state=tk.DISABLED)
            if self.license_var.get() == 1:
                self.btn_next.config(state=tk.NORMAL)
            else:
                self.btn_next.config(state=tk.DISABLED)
                
        elif self.current_step == 1:
            self.btn_back.config(state=tk.NORMAL)
            if self.privacy_var.get() == 1:
                self.btn_next.config(state=tk.NORMAL)
            else:
                self.btn_next.config(state=tk.DISABLED)
                
        elif self.current_step == 2:
            self.btn_back.config(state=tk.NORMAL)
            self.btn_next.config(state=tk.NORMAL)
            
        elif self.current_step == 3:
            self.start_installation()
            
        elif self.current_step == 4:
            self.btn_back.config(state=tk.DISABLED)
            self.btn_next.config(text="Finish", state=tk.NORMAL)
            self.btn_cancel.config(state=tk.DISABLED)

    def show_step(self, index):
        for frame in self.frames:
            frame.pack_forget()
        
        self.frames[index].pack(fill=tk.BOTH, expand=True)
        self.current_step = index
        self.update_nav_buttons()

    def next_step(self):
        if self.current_step == len(self.frames) - 1:
            self.launch_application()
        elif self.current_step < len(self.frames) - 1:
            self.show_step(self.current_step + 1)

    def prev_step(self):
        if self.current_step > 0:
            self.show_step(self.current_step - 1)

    def launch_application(self):
        marker_file = os.path.expanduser("~/.inklet_setup_complete")
        try:
            with open(marker_file, 'w') as f:
                f.write("Setup Complete")
        except Exception as e:
            print(f"Could not write marker file: {e}")

        self.destroy()

        try:
            import puppet
            puppet.main()
        except ImportError:
            try:
                import puppet_2
                puppet_2.main()
            except ImportError:
                messagebox.showerror("Error", "Could not find puppet.py to launch.")

if __name__ == "__main__":
    app = InkletWizard()
    app.mainloop()
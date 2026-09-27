#!/usr/bin/env python3
import os
import sys
import platform
import traceback

# ==========================================
# CROSS-PLATFORM DPI & ENVIRONMENT SETUP
# ==========================================
# Ensure crisp rendering on High-DPI/Retina displays for PySide6
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

# Ensure crisp rendering for Tkinter on Windows
if platform.system() == 'Windows':
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

def show_crash_dialog(title, message):
    """Displays a native error popup if the app crashes in hidden-console mode."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        print(f"{title}: {message}")

def main():
    try:
        # Universal home directory marker file (safely resolves on Mac, Windows, and Linux)
        marker_file = os.path.expanduser("~/.inklet_setup_complete")
        
        if not os.path.exists(marker_file):
            # 1. First time running: Import and launch the Setup Wizard
            import wizard
            app = wizard.InkletWizard()
            app.mainloop()
        else:
            # 2. Already installed: Import and launch the Main 3D Puppet
            import puppet
            puppet.main()
            
    except Exception as e:
        # Catch all fatal errors and display them so the user isn't left guessing
        error_trace = traceback.format_exc()
        show_crash_dialog(
            "Inklet Critical Error", 
            f"The application encountered a fatal error and cannot continue:\n\n{str(e)}\n\n{error_trace}"
        )
        sys.exit(1)

if __name__ == "__main__":
    # Ensure PyInstaller's temporary unpack directory is first in the module search path
    if getattr(sys, 'frozen', False):
        sys.path.insert(0, sys._MEIPASS)
        
    main()
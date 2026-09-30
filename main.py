import sys
# Prevent OpenCV binary extension recursive loader bug in PyInstaller
sys.OpenCV_REPLACE_SYS_PATH_0 = True
import os
import platform
import traceback
import multiprocessing

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

def should_show_wizard():
    """Determines whether the setup wizard should be presented."""
    current_exe = os.path.abspath(sys.executable)
    # 1. Always show wizard if running directly from the mounted installer DMG volume
    if "/Volumes/" in current_exe or "/Volumes/" in os.path.abspath(sys.argv[0]):
        return True

    marker_file = os.path.expanduser("~/.inklet_setup_complete")
    if not os.path.exists(marker_file):
        return True

    # 2. Check if the app installation recorded matches the current installation
    try:
        with open(marker_file, "r") as f:
            saved_info = f.read().strip()
        
        # If running as a frozen app bundle (.app)
        app_bundle = current_exe
        while app_bundle and not app_bundle.endswith(".app") and app_bundle != "/":
            app_bundle = os.path.dirname(app_bundle)
            
        if app_bundle.endswith(".app") and os.path.exists(app_bundle):
            st = os.stat(app_bundle)
            current_id = f"{app_bundle}:{st.st_ino}:{int(getattr(st, 'st_birthtime', st.st_ctime))}"
            if saved_info != current_id:
                # App was reinstalled, replaced, or run from a new location
                return True
            return False
        else:
            return False
    except Exception:
        return True

def main():
    try:
        if should_show_wizard():
            # 1. First time running / re-installed / running from DMG: Launch Setup Wizard
            import wizard
            app = wizard.InkletWizard()
            app.mainloop()
        else:
            # 2. Already installed and setup complete: Launch Main 3D Puppet
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
    multiprocessing.freeze_support()
    # Ensure PyInstaller's temporary unpack directory is first in the module search path
    if getattr(sys, 'frozen', False):
        sys.path.insert(0, sys._MEIPASS)
        try:
            os.chdir(sys._MEIPASS)
        except Exception:
            pass
        
    main()
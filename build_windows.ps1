# ==============================================================================
# INKLET - Windows build: PyInstaller onedir -> .exe + Inno Setup installer
# ==============================================================================
# Run on Windows in PowerShell (CANNOT be built from macOS/Linux):
#   python -m venv venv; .\venv\Scripts\Activate.ps1
#   pip install -r requirements.txt
#   pip uninstall -y jax jaxlib scipy        # slim: not needed by Tasks API
#   .\build_windows.ps1
# Produces:
#   dist\Inklet\Inklet.exe                       (portable folder)
#   dist\Inklet-1.0.0-windows-x64-portable.zip
#   dist\Inklet-1.0.0-windows-x64-setup.exe      (if Inno Setup installed)
# ==============================================================================
$ErrorActionPreference = "Stop"
$AppName = "Inklet"
$Version = "1.0.0"

Write-Host "=== Generating .ico icon ===" -ForegroundColor Cyan
python -c "from PIL import Image; Image.open('cap.png').convert('RGBA').resize((256,256),Image.NEAREST).save('app_icon.ico', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
$env:INKLET_ICON = (Resolve-Path "app_icon.ico").Path

Write-Host "=== PyInstaller ===" -ForegroundColor Cyan
if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path "dist\$AppName") { Remove-Item -Recurse -Force "dist\$AppName" }
python -m PyInstaller inklet.spec --noconfirm --clean

$exePath = "dist\$AppName\$AppName.exe"
if (-not (Test-Path $exePath)) { throw "Build produced no $exePath" }

# ---- Portable ZIP ----
$zipOut = "dist\$AppName-$Version-windows-x64-portable.zip"
if (Test-Path $zipOut) { Remove-Item -Force $zipOut }
Compress-Archive -Path "dist\$AppName\*" -DestinationPath $zipOut
Write-Host ">> $zipOut" -ForegroundColor Green

# ---- Inno Setup installer ----
$iscc = $null
foreach ($p in @(
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe")) {
  if (Test-Path $p) { $iscc = $p; break }
}
if ($iscc) {
  Write-Host "=== Inno Setup installer ===" -ForegroundColor Cyan
  & $iscc "installer_windows.iss"
  Write-Host ">> dist\$AppName-$Version-windows-x64-setup.exe" -ForegroundColor Green
} else {
  Write-Host "!! Inno Setup (ISCC.exe) not found -> skipping setup.exe." -ForegroundColor Yellow
  Write-Host "   Install from https://jrsoftware.org/isdl.php then re-run." -ForegroundColor Yellow
}

Remove-Item -Force app_icon.ico -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Windows build complete. Artifacts in .\dist" -ForegroundColor Cyan
Get-ChildItem dist\*.zip, dist\*setup.exe -ErrorAction SilentlyContinue | Select-Object Name

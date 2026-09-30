#!/bin/bash
# ==============================================================================
# INKLET - Linux build: PyInstaller onedir -> .tar.gz + AppImage
# ==============================================================================
# Run on Linux. Slim tip: pip uninstall -y jax jaxlib scipy before building.
# ==============================================================================
set -euo pipefail
APP_NAME="Inklet"; VERSION="1.0.0"; ARCH="$(uname -m)"

python3 -c "from PIL import Image; Image.open('cap.png').convert('RGBA').resize((256,256),Image.NEAREST).save('inklet.png')"
export INKLET_ICON="$PWD/inklet.png"

rm -rf build "dist/$APP_NAME"
python3 -m PyInstaller inklet.spec --noconfirm --clean

APPDIR_SRC="dist/$APP_NAME"
[ -d "$APPDIR_SRC" ] || { echo "!! build produced no dist/$APP_NAME"; exit 1; }

TAR_OUT="dist/${APP_NAME}-${VERSION}-linux-${ARCH}.tar.gz"
rm -f "$TAR_OUT"; tar -C dist -czf "$TAR_OUT" "$APP_NAME"; echo ">> $TAR_OUT"

APPDIR="dist/${APP_NAME}.AppDir"; rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" "$APPDIR/usr/share/icons/hicolor/256x256/apps"
cp -R "$APPDIR_SRC/." "$APPDIR/usr/bin/"
cp inklet.png "$APPDIR/inklet.png"
cp inklet.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/inklet.png"
cat > "$APPDIR/inklet.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$APP_NAME
Exec=$APP_NAME
Icon=inklet
Categories=Graphics;Utility;
Terminal=false
EOF
cp "$APPDIR/inklet.desktop" "$APPDIR/usr/share/applications/inklet.desktop"
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "${0}")")"
exec "$HERE/usr/bin/Inklet" "$@"
EOF
chmod +x "$APPDIR/AppRun"

if command -v appimagetool >/dev/null 2>&1; then
  APPIMG_OUT="dist/${APP_NAME}-${VERSION}-linux-${ARCH}.AppImage"
  ARCH="$ARCH" appimagetool "$APPDIR" "$APPIMG_OUT"; echo ">> $APPIMG_OUT"
else
  echo "!! appimagetool not found -> AppImage skipped (tar.gz still produced)."
fi
rm -f inklet.png
echo "Linux build complete."; ls -1 dist/*.tar.gz dist/*.AppImage 2>/dev/null || true

#!/bin/bash
# ==============================================================================
# INKLET - macOS build: .app -> .dmg + .pkg  (arm64 and/or x86_64)
# ==============================================================================
# Usage:
#   ./build_mac.sh              # build for the host arch
#   ./build_mac.sh arm64        # Apple Silicon only
#   ./build_mac.sh x86_64       # Intel only  (needs an x86_64 python via Rosetta)
#   ./build_mac.sh both         # both arches (needs both python interpreters)
# ==============================================================================
set -euo pipefail

APP_NAME="Inklet"
BUNDLE_ID="com.shoroz.inklet"
VERSION="1.0.0"
HOST_ARCH="$(uname -m)"
DIST_DIR="dist"

ARG="${1:-$HOST_ARCH}"
case "$ARG" in
  arm64)  ARCHES=(arm64) ;;
  x86_64) ARCHES=(x86_64) ;;
  both)   ARCHES=(arm64 x86_64) ;;
  *)      ARCHES=("$HOST_ARCH") ;;
esac

# --- Generate a proper .icns from cap.png -------------------------------------
make_icns() {
  local out="app_icon.icns"
  rm -rf inklet.iconset "$out"
  mkdir -p inklet.iconset
  python3 - "$out" <<'PY'
import sys, os
from PIL import Image
src = "cap.png"
img = Image.open(src).convert("RGBA")
sizes = [16,32,64,128,256,512]
os.makedirs("inklet.iconset", exist_ok=True)
for s in sizes:
    img.resize((s,s), Image.NEAREST).save(f"inklet.iconset/icon_{s}x{s}.png")
    img.resize((s*2,s*2), Image.NEAREST).save(f"inklet.iconset/icon_{s}x{s}@2x.png")
PY
  if command -v iconutil >/dev/null 2>&1; then
    iconutil -c icns inklet.iconset -o "$out"
  else
    python3 -c "from PIL import Image; Image.open('cap.png').convert('RGBA').resize((512,512),Image.NEAREST).save('$out')"
  fi
  rm -rf inklet.iconset
  echo "$PWD/$out"
}

# --- Pick a python interpreter for a given arch -------------------------------
python_for_arch() {
  local arch="$1"
  if [ "$arch" = "$HOST_ARCH" ]; then
    echo "python3"; return 0
  fi
  # Cross-arch: try Rosetta on Apple Silicon hosts building x86_64.
  if [ "$HOST_ARCH" = "arm64" ] && [ "$arch" = "x86_64" ]; then
    if /usr/bin/arch -x86_64 /usr/bin/true 2>/dev/null; then
      echo "arch -x86_64 python3"; return 0
    fi
  fi
  return 1
}

echo "=== Generating icon ==="
ICON_PATH="$(make_icns)"
export INKLET_ICON="$ICON_PATH"

for ARCH in "${ARCHES[@]}"; do
  echo ""
  echo "############################################################"
  echo "# Building $APP_NAME for macOS $ARCH"
  echo "############################################################"

  if ! PY_CMD="$(python_for_arch "$ARCH")"; then
    echo "!! No $ARCH python available on this host; skipping $ARCH."
    echo "   (Install an $ARCH python, or run this script on an $ARCH mac.)"
    continue
  fi

  export INKLET_TARGET_ARCH="$ARCH"
  rm -rf build "dist/$APP_NAME" "dist/$APP_NAME.app"

  echo "--- PyInstaller ($PY_CMD) ---"
  $PY_CMD -m PyInstaller inklet.spec --noconfirm --clean

  APP_PATH="dist/$APP_NAME.app"
  [ -d "$APP_PATH" ] || { echo "!! Build produced no .app"; exit 1; }

  echo "--- Ad-hoc codesign ---"
  codesign --force --deep --sign - "$APP_PATH"

  STAGE="dist/stage_${ARCH}"
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  cp -R "$APP_PATH" "$STAGE/"

  # ---- DMG ----
  DMG_OUT="dist/${APP_NAME}-${VERSION}-mac-${ARCH}.dmg"
  rm -f "$DMG_OUT"
  echo "--- DMG ---"
  if command -v create-dmg >/dev/null 2>&1 && create-dmg \
        --volname "$APP_NAME" \
        --window-pos 200 120 --window-size 500 350 --icon-size 80 \
        --icon "$APP_NAME.app" 130 170 --hide-extension "$APP_NAME.app" \
        --app-drop-link 370 170 --overwrite \
        "$DMG_OUT" "$STAGE"; then
    echo "create-dmg ok"
  else
    echo "create-dmg unavailable/failed -> hdiutil fallback"
    ln -sf /Applications "$STAGE/Applications"
    hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -ov -format UDZO "$DMG_OUT"
  fi

  # ---- PKG ----
  PKG_OUT="dist/${APP_NAME}-${VERSION}-mac-${ARCH}.pkg"
  rm -f "$PKG_OUT"
  echo "--- PKG ---"
  PKG_ROOT="dist/pkgroot_${ARCH}"
  rm -rf "$PKG_ROOT"; mkdir -p "$PKG_ROOT/Applications"
  cp -R "$APP_PATH" "$PKG_ROOT/Applications/"
  pkgbuild --root "$PKG_ROOT" \
           --identifier "$BUNDLE_ID" \
           --version "$VERSION" \
           --install-location "/" \
           "$PKG_OUT"

  rm -rf "$STAGE" "$PKG_ROOT"
  echo ">> $DMG_OUT"
  echo ">> $PKG_OUT"
done

rm -f "$ICON_PATH"
echo ""
echo "=============================================================="
echo "macOS build complete. Artifacts in ./dist:"
ls -1 dist/*.dmg dist/*.pkg 2>/dev/null || true
echo "=============================================================="

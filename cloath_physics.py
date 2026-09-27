#!/usr/bin/env python3
"""
CLOTH PHYSICS & RENDERING — CAPE SUBSYSTEM
============================================
A structural/shear/bend-constrained Verlet cloth sim, sub-stepped for
stability, plus a pixel-art renderer that matches the body's low-res
raster + nearest-neighbor-upscale technique.

WHAT CHANGED FROM A BASIC GRID SPRING SYSTEM (accuracy):
- Sub-stepping (3x per call): forces + integration + constraint solve run
  in several smaller slices instead of one big one, so fast head motion
  or a hard flip impulse can't blow the simulation up between frames.
- Full structural + shear + bend constraint topology (Provot-style):
  structural constraints alone let a cloth grid collapse into a
  parallelogram under torque; shear constraints (the diagonals) are what
  actually stop that; bend constraints (skip-one neighbors, soft) stop
  it folding into a sharp zigzag while still allowing real drape.
  A basic "distance constraints on the grid only" cloth is missing shear
  resistance entirely — that's usually the single biggest accuracy gap.
- Geometry changes (zoom/scale) rescale existing point positions instead
  of snapping back to a flat rest pose, so resizing the pet doesn't pop
  the cape flat and re-drape it.
- A soft boundary keeps the hem from swinging up through the head.

WHAT'S NEW FOR "JUICE":
- Reactions are impulse-driven, not just force-driven: a flip start or a
  big smile crossing its threshold gives the cloth an instant kick
  (done the correct Verlet way, by offsetting a point's *old* position
  so the next integration step reads it as velocity) instead of a slow
  ease-in. That's the difference between a cape that "notices" you and
  one that just drifts.
- Turbulence amplitude is driven by a rolling "motion energy" estimate
  (drag speed + yaw/tilt rate), so the cloth stays calm at rest and
  gets visibly more alive the moment you actually move the pet — not
  randomly, in proportion to what you're doing.
- The collar row eases toward its target instead of teleporting to it,
  so even the "pinned" edge has a touch of give.
- The renderer flashes a brief warm highlight across the cape right
  after an impulse — the physical "pop" gets a visual payoff.

INTERFACE (unchanged, drop-in compatible):
    VerletClothCape(cols=3, rows=10)
    .step(dt, collar_w, hem_w, cape_h, scale, d_collar_x, d_collar_y,
          yaw, pitch, tilt_deg, anim_state, anim_timer, smile,
          is_flipping, rot_x)
    draw_pixelated_cape(painter, cape_sim, anchor_x, anchor_y,
                         head_rx, head_ry, scale, color_name)

A NOTE ON THE PALETTES:
BLOSSOM_X6 and CAPE_PALETTES below are rebuilt from what's actually
referenced elsewhere in the app (BLOSSOM_X6["DEEP_BERRY"]/["BLACK"] in
the body renderer; cape colors Red/Blue #203671/Green #3fac95/
Purple #4f3a54 from the wardrobe rack). Blue/green/purple use those
exact hexes. Red and the rest of BLOSSOM_X6 are my best-guess fill —
swap them for your real values if they differ.
"""

import math
import random
import time

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QImage, QPainter, QPolygonF


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------

BLOSSOM_X6 = {
    "BLACK": "#0b0710",
    "WHITE": "#fff7fa",
    "DEEP_BERRY": "#4a0e28",
    "ROSE": "#b23a5e",
    "BLUSH": "#e8879f",
    "PETAL": "#f6c9d6",
}

CAPE_PALETTES = {
    "red":    {"base": "#c2273d", "light": "#e0536a", "dark": "#7a1526"},
    "blue":   {"base": "#203671", "light": "#3d55a0", "dark": "#141f42"},
    "green":  {"base": "#3fac95", "light": "#6ecab7", "dark": "#236152"},
    "purple": {"base": "#4f3a54", "light": "#75587c", "dark": "#2c1f30"},
}


# ---------------------------------------------------------------------------
# Point primitive
# ---------------------------------------------------------------------------

class _Pt:
    """A single Verlet point. Velocity is implicit: (x - ox, y - oy)."""
    __slots__ = ("x", "y", "ox", "oy", "pinned", "ax", "ay")

    def __init__(self, x, y, pinned=False):
        self.x = x
        self.y = y
        self.ox = x
        self.oy = y
        self.pinned = pinned
        self.ax = 0.0
        self.ay = 0.0


# ---------------------------------------------------------------------------
# Cloth simulation
# ---------------------------------------------------------------------------

class VerletClothCape:
    """
    Multi-segment Verlet cloth, structural + shear + bend constrained,
    sub-stepped, with an impulse system for snappy reactions.

    Coordinate space is LOCAL to the collar anchor (roughly (0, 0) at the
    top-center), not screen space — step() only ever receives how far the
    anchor just moved (d_collar_x/d_collar_y), never an absolute position.
    That delta is treated as a fictitious-force input (exactly like being
    pushed back into your seat when a car accelerates): the renderer is
    the only place that adds the real anchor_x/anchor_y back in.
    """

    SUBSTEPS = 3
    ITER = 4
    DAMPING = 0.965
    AIR_DRAG = 0.030

    def __init__(self, cols=3, rows=10):
        self.cols = max(2, cols)
        self.rows = max(2, rows)
        self.grid = []
        self._built = False
        self._dims = None

        self._struct = []
        self._shear = []
        self._bend = []

        self._prev_yaw = 0.0
        self._prev_pitch = 0.0
        self._prev_tilt = 0.0
        self._prev_flip = False
        self._prev_smile = 0.0
        self._energy = 0.0
        self.last_impulse_t = 0.0

        self._t0 = time.time()
        self._phase = random.uniform(0.0, math.tau)

    # -- geometry --------------------------------------------------------

    def _rest_row_width(self, r, collar_w, hem_w):
        t = r / float(self.rows - 1)
        # slight ease so the taper feels less linear/mechanical
        return collar_w + (hem_w - collar_w) * (t ** 1.15)

    def _build_flat(self, collar_w, hem_w, cape_h):
        self.grid = []
        row_h = cape_h / float(self.rows - 1)
        for r in range(self.rows):
            w = self._rest_row_width(r, collar_w, hem_w)
            row = []
            for c in range(self.cols):
                t = (c / float(self.cols - 1)) - 0.5
                row.append(_Pt(t * w, r * row_h, pinned=(r == 0)))
            self.grid.append(row)

    def _recompute_rest_lengths(self, collar_w, hem_w, cape_h):
        """Rest lengths always come from the IDEAL flat pose at the current
        dims, never from wherever the points currently happen to be — the
        constraint solver should pull toward the correct drape, not lock
        in whatever shape it finds."""
        row_h = cape_h / float(self.rows - 1)
        ideal = []
        for r in range(self.rows):
            w = self._rest_row_width(r, collar_w, hem_w)
            ideal.append([((c / float(self.cols - 1)) - 0.5) * w for c in range(self.cols)])

        def d(r1, c1, r2, c2):
            dx = ideal[r2][c2] - ideal[r1][c1]
            dy = (r2 - r1) * row_h
            return max(0.5, math.hypot(dx, dy))

        self._struct, self._shear, self._bend = [], [], []
        for r in range(self.rows):
            for c in range(self.cols):
                if c + 1 < self.cols:
                    self._struct.append([r, c, r, c + 1, d(r, c, r, c + 1)])
                if r + 1 < self.rows:
                    self._struct.append([r, c, r + 1, c, d(r, c, r + 1, c)])
                if r + 1 < self.rows and c + 1 < self.cols:
                    self._shear.append([r, c, r + 1, c + 1, d(r, c, r + 1, c + 1)])
                if r + 1 < self.rows and c - 1 >= 0:
                    self._shear.append([r, c, r + 1, c - 1, d(r, c, r + 1, c - 1)])
                if r + 2 < self.rows:
                    self._bend.append([r, c, r + 2, c, d(r, c, r + 2, c)])
                if c + 2 < self.cols:
                    self._bend.append([r, c, r, c + 2, d(r, c, r, c + 2)])

    def _dims_changed(self, collar_w, hem_w, cape_h):
        if self._dims is None:
            return True
        pc, ph, pcp = self._dims
        return (abs(pc - collar_w) > 1.0 or abs(ph - hem_w) > 1.0
                or abs(pcp - cape_h) > 1.0)

    def _rebuild(self, collar_w, hem_w, cape_h):
        if not self._built:
            self._build_flat(collar_w, hem_w, cape_h)
        else:
            # rescale existing (possibly deformed) points instead of
            # popping back to a flat rest pose on every zoom tick
            _, _, old_h = self._dims
            ratio = cape_h / max(1.0, old_h)
            for row in self.grid:
                for pt in row:
                    pt.x *= ratio
                    pt.y *= ratio
                    pt.ox *= ratio
                    pt.oy *= ratio
        self._recompute_rest_lengths(collar_w, hem_w, cape_h)
        self._dims = (collar_w, hem_w, cape_h)
        self._built = True

    def _collar_targets(self, collar_w, yaw, pitch, tilt_deg):
        targets = []
        tilt_rad = math.radians(tilt_deg)
        cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
        for c in range(self.cols):
            t = (c / float(self.cols - 1)) - 0.5
            bx, by = t * collar_w, 0.0
            rx = bx * cos_t - by * sin_t
            ry = bx * sin_t + by * cos_t
            rx += yaw * 4.0 * t
            ry += abs(t) * pitch * 3.0
            targets.append((rx, ry))
        return targets

    # -- reactions ---------------------------------------------------------

    def _impulse(self, kind, strength, rot_x=0.0):
        for r in range(1, self.rows):
            depth = r / float(self.rows - 1)
            for pt in self.grid[r]:
                if kind == "spin":
                    ang = math.radians(rot_x) + pt.x * 0.01
                    vx = math.sin(ang) * strength * depth
                    vy = -abs(math.cos(ang)) * strength * 0.3 * depth
                else:  # "puff"
                    side = 1.0 if pt.x >= 0 else -1.0
                    vx = side * strength * depth
                    vy = -strength * 0.4 * depth
                # inject velocity the Verlet way: push the OLD position
                # backward so the next integration step reads it as speed
                pt.ox = pt.x - vx * 0.016
                pt.oy = pt.y - vy * 0.016

    def _apply_forces(self, dt, scale, d_collar_x, d_collar_y, smile,
                       is_flipping, rot_x, anim_state, anim_timer, now):
        g = 640.0 * scale

        anim_boost = 0.0
        if anim_state == "GUST":
            anim_boost = min(1.0, anim_timer / 0.8) * 3.0
        elif anim_state == "FLUTTER":
            anim_boost = (0.5 + 0.5 * math.sin(anim_timer * 14.0)) * 1.6
        elif anim_state == "BILLOW":
            anim_boost = min(1.0, anim_timer / 1.2) * 2.2

        turb_amp = (2.0 + self._energy * 5.0 + anim_boost) * scale
        age = now - self._t0
        wind = ((math.sin(age * 2.1 + self._phase)
                 + 0.5 * math.sin(age * 4.7 + self._phase * 1.7)
                 + 0.22 * math.sin(age * 9.3 + self._phase * 2.3))
                * turb_amp)

        inertia_x = -d_collar_x * 5.5
        inertia_y = -d_collar_y * 5.5
        smile_lift = max(0.0, smile - 0.2) * 2.2 * scale
        flip_swirl = (math.sin(math.radians(rot_x)) * 7.0 * scale) if is_flipping else 0.0

        for r in range(1, self.rows):
            t = r / float(self.rows - 1)
            reach = 0.35 + 0.65 * t  # hem feels more of everything than the collar
            for pt in self.grid[r]:
                pt.ax = wind * reach + inertia_x * reach + flip_swirl * t
                pt.ay = g + inertia_y * reach * 0.6 - smile_lift * (1.0 - t * 0.4)

    def _integrate(self, dt):
        for row in self.grid:
            for pt in row:
                if pt.pinned:
                    continue
                vx = (pt.x - pt.ox) * self.DAMPING
                vy = (pt.y - pt.oy) * self.DAMPING
                speed = math.hypot(vx, vy)
                if speed > 1e-6:
                    drag = min(speed, self.AIR_DRAG * speed * speed / max(0.05, speed))
                    vx -= vx / speed * drag
                    vy -= vy / speed * drag
                nx = pt.x + vx + pt.ax * dt * dt
                ny = pt.y + vy + pt.ay * dt * dt
                pt.ox, pt.oy = pt.x, pt.y
                pt.x, pt.y = nx, ny

    def _solve(self, constraints, stiffness):
        g = self.grid
        for con in constraints:
            r1, c1, r2, c2, rest = con
            p1, p2 = g[r1][c1], g[r2][c2]
            dx, dy = p2.x - p1.x, p2.y - p1.y
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            diff = (dist - rest) / dist
            if dist < rest * 0.55:
                diff *= 0.35  # soft compression floor: allow bunching, don't fight it hard
            w1 = 0.0 if p1.pinned else 1.0
            w2 = 0.0 if p2.pinned else 1.0
            wsum = w1 + w2
            if wsum <= 0.0:
                continue
            cx, cy = dx * diff * stiffness, dy * diff * stiffness
            p1.x += cx * (w1 / wsum)
            p1.y += cy * (w1 / wsum)
            p2.x -= cx * (w2 / wsum)
            p2.y -= cy * (w2 / wsum)

    def _enforce_boundary(self, scale):
        limit_y = -18.0 * scale
        for row in self.grid[1:]:
            for pt in row:
                if pt.y < limit_y:
                    pt.y = limit_y
                    pt.oy = pt.y

    # -- public ------------------------------------------------------------

    def step(self, dt, collar_w, hem_w, cape_h, scale, d_collar_x, d_collar_y,
             yaw, pitch, tilt_deg, anim_state, anim_timer, smile,
             is_flipping, rot_x):

        if not self._built or self._dims_changed(collar_w, hem_w, cape_h):
            self._rebuild(collar_w, hem_w, cape_h)

        now = time.time()
        dt = max(1e-4, min(0.05, dt))

        d_yaw = yaw - self._prev_yaw
        d_tilt = tilt_deg - self._prev_tilt
        flip_started = is_flipping and not self._prev_flip
        smile_popped = smile > 0.35 and self._prev_smile <= 0.35

        drag_speed = math.hypot(d_collar_x, d_collar_y)
        motion = drag_speed * 3.0 + abs(d_yaw) * 30.0 + abs(d_tilt) * 0.5
        self._energy += (min(3.0, motion) - self._energy) * 0.25

        if flip_started:
            self._impulse("spin", 9.5 * scale, rot_x=rot_x)
            self.last_impulse_t = now
        if smile_popped:
            self._impulse("puff", 3.2 * scale)
            self.last_impulse_t = now

        targets = self._collar_targets(collar_w, yaw, pitch, tilt_deg)
        for c, pt in enumerate(self.grid[0]):
            tx, ty = targets[c]
            pt.x += (tx - pt.x) * 0.55
            pt.y += (ty - pt.y) * 0.55
            pt.ox, pt.oy = pt.x, pt.y

        sub_dt = dt / self.SUBSTEPS
        for _ in range(self.SUBSTEPS):
            self._apply_forces(sub_dt, scale, d_collar_x, d_collar_y, smile,
                                is_flipping, rot_x, anim_state, anim_timer, now)
            self._integrate(sub_dt)
            for _ in range(self.ITER):
                self._solve(self._struct, 1.0)
                self._solve(self._shear, 0.85)
                self._solve(self._bend, 0.22)
            self._enforce_boundary(scale)

        self._prev_yaw, self._prev_pitch, self._prev_tilt = yaw, pitch, tilt_deg
        self._prev_flip = is_flipping
        self._prev_smile = smile

    def iter_quads(self):
        """Yield (r, c, top_left, top_right, bottom_left, bottom_right) per panel."""
        g = self.grid
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                yield r, c, g[r][c], g[r][c + 1], g[r + 1][c], g[r + 1][c + 1]

    def boundary_points(self):
        """Outer silhouette as one closed loop: left edge down, hem across,
        right edge up, collar back across."""
        g = self.grid
        pts = [g[r][0] for r in range(self.rows)]
        pts += [g[self.rows - 1][c] for c in range(1, self.cols)]
        pts += [g[r][self.cols - 1] for r in range(self.rows - 2, -1, -1)]
        pts += [g[0][c] for c in range(self.cols - 2, 0, -1)]
        return pts


# ---------------------------------------------------------------------------
# Rendering — same low-res-buffer + nearest-neighbor-upscale trick as the
# body, so the cape doesn't visually clash with the rest of the character.
# ---------------------------------------------------------------------------

def draw_pixelated_cape(painter, cape_sim, anchor_x, anchor_y, head_rx, head_ry,
                         scale, color_name="red"):
    if not getattr(cape_sim, "_built", False) or not cape_sim.grid:
        return

    pal = CAPE_PALETTES.get(color_name, CAPE_PALETTES["red"])
    px = max(2, int(round(2.6 * scale)))

    xs = [pt.x for row in cape_sim.grid for pt in row]
    ys = [pt.y for row in cape_sim.grid for pt in row]
    margin = px * 3
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    pw = max(8, int(max_x - min_x) // px)
    ph = max(8, int(max_y - min_y) // px)

    def to_buf(x, y):
        return QPointF((x - min_x) / px, (y - min_y) / px)

    boundary_poly = QPolygonF([to_buf(p.x, p.y) for p in cape_sim.boundary_points()])

    # 1. Dilated white silhouette drawn first (background) — the sliver
    #    that peeks out from under the panel fill becomes the outline.
    mask = QImage(pw, ph, QImage.Format.Format_ARGB32_Premultiplied)
    mask.fill(Qt.GlobalColor.transparent)
    mp = QPainter(mask)
    mp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    mp.setPen(Qt.PenStyle.NoPen)
    mp.setBrush(QColor(255, 255, 255))
    mp.drawPolygon(boundary_poly)
    mp.end()

    buf = QImage(pw, ph, QImage.Format.Format_ARGB32_Premultiplied)
    buf.fill(Qt.GlobalColor.transparent)
    bp = QPainter(buf)
    bp.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    bp.setPen(Qt.PenStyle.NoPen)

    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]:
        bp.drawImage(dx, dy, mask)

    # 2. Filled panels on top, shaded per-quad by local fold/shear so the
    #    cape reads as folded fabric rather than a flat cutout.
    base_c, light_c, dark_c = QColor(pal["base"]), QColor(pal["light"]), QColor(pal["dark"])
    for _r, _c, tl, tr, bl, br in cape_sim.iter_quads():
        top_dx, bot_dx = tr.x - tl.x, br.x - bl.x
        span = max(1.0, (abs(top_dx) + abs(bot_dx)) * 0.5)
        shade = max(-1.0, min(1.0, (bot_dx - top_dx) / span))
        col = light_c if shade > 0.14 else (dark_c if shade < -0.14 else base_c)
        quad = QPolygonF([to_buf(tl.x, tl.y), to_buf(tr.x, tr.y),
                           to_buf(br.x, br.y), to_buf(bl.x, bl.y)])
        bp.setBrush(col)
        bp.drawPolygon(quad)

    # 3. Brief warm flash right after an impulse — the visual payoff for
    #    the physics "pop".
    since_impulse = time.time() - getattr(cape_sim, "last_impulse_t", 0.0)
    if since_impulse < 0.18:
        fade = 1.0 - (since_impulse / 0.18)
        glow = QColor(255, 235, 190, int(130 * fade))
        bp.setBrush(glow)
        bp.drawPolygon(boundary_poly)

    bp.end()

    dest_w, dest_h = pw * px, ph * px
    tuck_y = head_ry * 0.08  # nudge slightly further behind the head silhouette
    dest_rect = QRectF(anchor_x + min_x, anchor_y + min_y - tuck_y, dest_w, dest_h)

    prev_smooth = painter.renderHints() & QPainter.RenderHint.SmoothPixmapTransform
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.drawImage(dest_rect, buf)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, bool(prev_smooth))
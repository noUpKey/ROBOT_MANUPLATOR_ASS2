#!/usr/bin/env python3
"""
AR523: Robot Manipulators -- Assignment 2
Force-Controlled Writing & Haptic Material Identification
"""

import argparse
import itertools
import os
import time

import numpy as np
import pybullet as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import *


# ==============================================================================
# PART 1: Trajectory Generation
# ==============================================================================

# run_writing_task() drives every polyline edge as its own rest-to-rest min-jerk
# move lasting at least 0.15 s, so a dense point cloud would crawl and stop at
# every point. Patterns are therefore emitted as sparse polygons: the fewest
# vertices that stay within PATTERN_TOL * scale of the ideal curve (0.4 mm for
# the default 10 cm pattern, against a 12 mm tool radius).
PATTERN_TOL = 0.004
PATTERN_SAMPLES = 2001      # dense samples per curve (odd, so t = pi is sampled exactly)


def _simplify(pts, tol):
    """Douglas-Peucker: keep the fewest vertices of the dense (N, 2) polyline
    `pts` such that every dropped point is within `tol` of the new polygon.
    Distances are measured to the edge itself, not its infinite line, so cusps
    and hairpins (e.g. the tips of the heart) always survive as vertices."""
    keep = np.zeros(len(pts), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        edge = pts[j] - pts[i]
        rel = pts[i + 1:j] - pts[i]
        len2 = edge @ edge      # zero for a closed curve: measure from the start point
        u = np.clip(rel @ edge / len2, 0.0, 1.0) if len2 > 0 else np.zeros(len(rel))
        dist = np.linalg.norm(rel - u[:, None] * edge, axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            k += i + 1
            keep[k] = True
            stack += [(i, k), (k, j)]
    return pts[keep]


def generate_pattern(shape, scale=0.1):
    """
    Generate the pen-down path of a pattern.

    shape: 'circle', 'spiral', 'infinity' or 'heart'.
    scale: overall size in metres. Each pattern is centred on the origin of the
           writing-plane frame (the spiral winds outward from it) and fits in a
           scale x scale square (circle: diameter, spiral: outer diameter,
           infinity / heart: width).

    Returns a list of strokes (each stroke is a list of (x, y) tuples, in
    metres, x = right, y = up). Every pattern is a single continuous stroke, so
    the pen touches down once; closed shapes end exactly where they start.
    """
    r = 0.5 * scale                                     # half-size of the pattern
    t = np.linspace(0.0, 2.0 * np.pi, PATTERN_SAMPLES)
    name = str(shape).strip().lower()

    if name == "circle":
        # counter-clockwise from 3 o'clock
        x, y = r * np.cos(t), r * np.sin(t)
    elif name == "spiral":
        # Archimedean spiral: radius grows linearly with angle, 3 turns, starting
        # at 15% of the outer radius (a tighter curl than the tool is worth drawing)
        turns = 3
        rad = r * (0.15 + 0.85 * t / t[-1])
        x, y = rad * np.cos(turns * t), rad * np.sin(turns * t)
    elif name == "infinity":
        # lemniscate of Bernoulli, tips at (+-r, 0), crossing at the origin
        d = 1.0 + np.sin(t) ** 2
        x, y = r * np.cos(t) / d, r * np.sin(t) * np.cos(t) / d
    elif name == "heart":
        # classic heart curve with cusps at the top notch (t = 0) and bottom tip
        # (t = pi); width normalised to 2r, bounding box centred on the origin
        x = 16.0 * np.sin(t) ** 3
        y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
        x, y = x * r / 16.0, (y - 0.5 * (y.max() + y.min())) * r / 16.0
    else:
        raise ValueError(f"Unknown pattern {shape!r}; expected 'circle', 'spiral', 'infinity' or 'heart'.")

    pts = np.column_stack([x, y])
    if name != "spiral":
        pts[-1] = pts[0]                                # closed curve: end exactly at the start
    pts = _simplify(pts, PATTERN_TOL * abs(scale))
    return [[(float(px), float(py)) for px, py in pts]]


# ==============================================================================
# PART 2: Cartesian Impedance Control Law
# ==============================================================================

class CartesianImpedanceController:

    def __init__(self, arm):
        self.arm = arm
        self.q_ref = None           # posture the redundant 7th DoF is held near (set on first step)

    def step(self, pos_des, quat_des, K_trans, K_rot, damping_ratio=1.0, vel_des=None):
        # ----------------------------------------------------------------------
        # TODO 2: Implement the Cartesian impedance torque law.
        # Inputs:
        #   pos_des: (3,) desired position
        #   quat_des: (4,) desired orientation quaternion
        #   K_trans: (3,) translational stiffness (x,y,z)
        #   K_rot: (3,) rotational stiffness (x,y,z)
        #   damping_ratio: scalar
        #   vel_des: (3,) desired linear velocity (optional, default zero)
        # Returns:
        #   tau: (7,) joint torques
        #   info: dict for logging (provided in solution)
        # ----------------------------------------------------------------------
        arm = self.arm
        q, qdot = arm.get_joint_state()
        pos, orn, _, _ = arm.get_tool_pose()
        if self.q_ref is None:
            self.q_ref = q.copy()

        J = arm.jacobian(q, qdot)                      # (6, 7) at the tool tip, world frame
        xdot = J @ qdot                                # tool twist [v; w]

        # Task-space stiffness and critical-style damping B = 2 * zeta * sqrt(K * M),
        # using the reflected mass / inertia constants from utils.
        K = np.concatenate([np.asarray(K_trans, float), np.asarray(K_rot, float)])
        M = np.concatenate([np.full(3, REFLECTED_MASS_TRANS), np.full(3, REFLECTED_INERTIA_ROT)])
        B = 2.0 * damping_ratio * np.sqrt(K * M)

        # Pose error with the same sign convention (current - desired) for both parts.
        e = np.concatenate([pos - np.asarray(pos_des, float), quat_to_axis_angle_error(orn, quat_des)])

        # The damper acts on the velocity *error*: damping the absolute velocity of a tip
        # that should be moving drags it B v / K behind its setpoint (~14 mm at 4 cm/s
        # with the writing gains). Without vel_des this is the plain  -B xdot  damper.
        xdot_des = np.zeros(6)
        if vel_des is not None:
            xdot_des[:3] = vel_des

        # Impedance law  F = -K e - B (xdot - xdot_des),  mapped to joints with J^T, plus
        # gravity compensation so the spring only has to act on the pose error.
        F = -K * e - B * (xdot - xdot_des)
        tau = J.T @ F + arm.gravity_compensation(q)

        # 7 joints for a 6-D task: damp and hold the redundant motion in the null space
        # so it cannot drift (it does not disturb the tool pose).
        N = np.eye(arm.n_joints) - np.linalg.pinv(J) @ J
        tau = tau + N @ (-5.0 * (q - self.q_ref) - 2.0 * qdot)

        info = dict(pos_error=e[:3], rot_error=e[3:], force_cmd=F[:3], tool_pos=pos, tool_vel=xdot[:3])
        return tau, info


# ==============================================================================
# PART 3: Admittance Control Outer Loop
# ==============================================================================

class AdmittanceController:
    def __init__(self, mass=ADMITTANCE_MASS, damping=ADMITTANCE_DAMPING):
        self.M = mass
        self.B = damping
        self.xc = 0.0
        self.xc_dot = 0.0

    def reset(self):
        self.xc = 0.0
        self.xc_dot = 0.0

    def step(self, f_measured, f_desired, dt, xc_limit=0.03, xc_dot_limit=0.08):
        """
        TODO 3: Implement admittance mass-damper outer loop step.
        Returns: float (self.xc)
        """
        # Virtual mass-damper driven by the force error:  M xc'' + B xc' = f_meas - f_des.
        # Pressing harder than desired pushes the setpoint offset xc up (away from the
        # surface); too light a touch lets it sink in. No spring term, so it integrates
        # the force error and settles exactly at f_meas = f_des.
        xc_ddot = (f_measured - f_desired - self.B * self.xc_dot) / self.M
        self.xc_dot = float(np.clip(self.xc_dot + xc_ddot * dt, -xc_dot_limit, xc_dot_limit))
        self.xc += self.xc_dot * dt
        if abs(self.xc) > xc_limit:                    # saturate the offset and stop winding up
            self.xc = float(np.sign(self.xc) * xc_limit)
            self.xc_dot = 0.0
        return self.xc

ADMITTANCE_NOMINAL_PENETRATION = 0.015

# Impedance writing renders the contact force with a virtual penetration (see
# run_writing_task). CONTACT_STIFFNESS_EST is the surface's stiffness in series with
# the controller's, and IMPEDANCE_RAMP_TIME softens the setpoint at each touch-down.
CONTACT_STIFFNESS_EST = 750.0    # N/m: tool and surface are both 1500 N/m (build_world), in series
IMPEDANCE_RAMP_TIME = 0.3        # s
IMPEDANCE_K_TANGENT = 1500.0    # N/m: impedance stiffness along the page (x, y)


# ==============================================================================
# LIVE VISUALIZATION (TODO 8)
# ==============================================================================

# The pen leaves ink only while it is actually pressing on the page. The marks are
# drawn a hair above the surface so they do not z-fight with its top face.
INK_FORCE_THRESHOLD = 0.5        # N
INK_COLOR = (0.05, 0.12, 0.45)   # dark blue ink (RGB, 0-1) for the PyBullet marks
INK_HEIGHT = 0.0008              # m

# Material-probe curve colours, close to each pad's colour in the simulation
# (A red-ish, B blue, C green-ish).
SAMPLE_COLORS = {"sample_A": "#eb6834", "sample_B": "#2a78d6", "sample_C": "#1baf7a"}


class LiveDashboard:
    """Real-time matplotlib dashboard (plt.ion()).

    mode="writing":  left  = desired path vs. the ink the tip actually leaves (top-down),
                     right = measured Fz vs. desired force, with the running RMSE.
    mode="material": left  = loading curve (force vs. measured penetration depth) of
                     every probed sample, with its least-squares slope,
                     right = force vs. time of each press.
    """
    REFRESH_PERIOD = 0.1         # s of wall-clock time between redraws
    PLOT_EVERY = 5               # plot every 5th physics step (200 Hz); metrics use all of them

    def __init__(self, mode="writing", f_desired=F_DESIRED, plan=None, t_total=None, title=None):
        self.mode = mode
        self.f_desired = f_desired
        self.last_update = 0.0
        self.n_samples = 0
        self.blit_artists, self.background = [], None
        # main.py / utils.py select the non-interactive "Agg" backend (needed for saving
        # figures headless); a live window needs a GUI backend, so switch for the dashboard.
        for backend in ("TkAgg", "QtAgg", "Qt5Agg"):
            try:
                plt.switch_backend(backend)
                break
            except Exception:
                continue
        else:
            print("[LiveDashboard] no interactive matplotlib backend found; nothing will be shown.")
        plt.ion()

        if mode == "writing":
            self._init_writing(plan or [], t_total, title)
        else:
            self._init_material(title)
        self.fig.tight_layout()
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._redraw(full=True)

    # ------------------------------------------------------------------ writing
    def _init_writing(self, plan, t_total, title):
        self.fig, (self.ax_xy, self.ax_f) = plt.subplots(
            1, 2, figsize=(13, 5.8), gridspec_kw=dict(width_ratios=[1, 1.3]))
        self.fig.suptitle(title or "Force-controlled writing", fontsize=13, weight="bold", color=NAVY)
        self.t_data, self.fz, self.fz_des = [], [], []
        self.ink_x, self.ink_y, self.air_x, self.air_y = [], [], [], []
        self.t_now = self.fz_now = self.tip_act = self.tip_des = None
        self.fz_peak = 0.0          # largest force so far, plotted or not (sets the y range)
        self.sq_err, self.n_err = 0.0, 0

        # Desired path: every pen-down move of the plan, broken (NaN) at pen lifts.
        xs, ys, prev_kind = [], [], None
        for kind, xy_from, xy_to in plan:
            if kind == "down":
                if prev_kind != "down":
                    xs.append(xy_from[0]); ys.append(xy_from[1])
                xs.append(xy_to[0]); ys.append(xy_to[1])
            else:
                xs.append(np.nan); ys.append(np.nan)
            prev_kind = kind

        ax = self.ax_xy
        ax.plot(xs, ys, color=GREY, ls="--", lw=1.2, label="desired path")
        self.line_air, = ax.plot([], [], color=TEAL, ls=":", lw=1.0, label="pen up (in air)")
        self.line_ink, = ax.plot([], [], color=NAVY, lw=2.2, solid_capstyle="round",
                                 label=f"ink: actual path (Fz > {INK_FORCE_THRESHOLD:g} N)")
        self.dot_des, = ax.plot([], [], "o", mfc="none", mec=GOLD, mew=2, ms=11, label="desired tip")
        self.dot_act, = ax.plot([], [], "o", color=RED, ms=5, label="actual tip")
        if np.isfinite(xs).any():
            margin = 1.5 * TOOL_RADIUS
            ax.set_xlim(np.nanmin(xs) - margin, np.nanmax(xs) + margin)
            ax.set_ylim(np.nanmin(ys) - margin, np.nanmax(ys) + margin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Desired vs. actual trajectory (top-down)", fontsize=11, color=NAVY)
        ax.legend(fontsize=8, frameon=False, loc="lower left")

        ax = self.ax_f
        ax.axhspan(self.f_desired - 0.5, self.f_desired + 0.5, color=GREY, alpha=0.10, lw=0, label="±0.5 N band")
        self.line_fdes, = ax.plot([], [], color=GREY, ls="--", lw=1.6, label="desired Fz")
        self.line_fz, = ax.plot([], [], color=RED, lw=1.1, label="measured Fz")
        ax.set_xlim(0.0, 1.02 * t_total if t_total else 10.0)
        ax.set_ylim(-0.3, 2.0 * self.f_desired + 0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normal force (N)")
        ax.set_title("Force tracking", fontsize=11, color=NAVY)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        self.status = ax.text(0.99, 0.98, "", transform=ax.transAxes, ha="right", va="top",
                              fontsize=10, family="monospace", color=NAVY)
        for ax in (self.ax_xy, self.ax_f):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color=GREY, alpha=0.15, lw=0.6)
        self._set_blit_artists([self.line_air, self.line_ink, self.dot_des, self.dot_act,
                                self.line_fdes, self.line_fz, self.status])

    def update_writing(self, t, pos_act, pos_des, f_act, f_des):
        fz = float(f_act[2])
        if f_des > 0:
            self.sq_err += (fz - f_des) ** 2
            self.n_err += 1
        self.t_now, self.fz_now, self.tip_act, self.tip_des = t, fz, pos_act, pos_des
        self.fz_peak = max(self.fz_peak, fz)
        self.n_samples += 1
        if self.n_samples % self.PLOT_EVERY == 0:
            inked = fz > INK_FORCE_THRESHOLD
            self.t_data.append(t)
            self.fz.append(fz)
            self.fz_des.append(f_des)
            self.ink_x.append(pos_act[0] if inked else np.nan)
            self.ink_y.append(pos_act[1] if inked else np.nan)
            self.air_x.append(np.nan if inked else pos_act[0])
            self.air_y.append(np.nan if inked else pos_act[1])
        self._redraw()

    def _refresh_writing(self):
        """Push the new data into the artists; True if the axes limits had to grow."""
        self.line_ink.set_data(self.ink_x, self.ink_y)
        self.line_air.set_data(self.air_x, self.air_y)
        self.line_fz.set_data(self.t_data, self.fz)
        self.line_fdes.set_data(self.t_data, self.fz_des)
        if self.t_now is None:
            return False
        self.dot_act.set_data([self.tip_act[0]], [self.tip_act[1]])
        self.dot_des.set_data([self.tip_des[0]], [self.tip_des[1]])
        rmse = f"{np.sqrt(self.sq_err / self.n_err):5.2f} N" if self.n_err else "    -"
        self.status.set_text(f"t    = {self.t_now:5.1f} s\nFz   = {self.fz_now:5.2f} N\nRMSE = {rmse}")
        # grow the force axes if the run outlasts / overshoots them
        grown = False
        x0, x1 = self.ax_f.get_xlim()
        if self.t_now > x1:
            self.ax_f.set_xlim(x0, 1.2 * self.t_now)
            grown = True
        y0, y1 = self.ax_f.get_ylim()
        if self.fz_peak > y1:
            self.ax_f.set_ylim(y0, 1.2 * self.fz_peak)
            grown = True
        return grown

    # ----------------------------------------------------------------- material
    def _init_material(self, title):
        self.fig, (self.ax_fd, self.ax_ft) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle(title or "Haptic material identification", fontsize=13, weight="bold", color=NAVY)
        self.name, self.color = None, NAVY
        self.depths, self.forces, self.times = [], [], []
        self.f_peak = 0.0           # largest force so far, plotted or not (sets the y range)
        self.results = []

        ax = self.ax_fd
        ax.set_xlabel("Penetration depth (mm)")
        ax.set_ylabel("Contact force (N)")
        ax.set_title("Loading curve: force vs. penetration", fontsize=11, color=NAVY)
        self.results_text = ax.text(0.98, 0.98, "", transform=ax.transAxes, ha="right", va="top",
                                    fontsize=9, family="monospace", color=NAVY)
        ax = self.ax_ft
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Contact force (N)")
        ax.set_title("Force during each press", fontsize=11, color=NAVY)
        # Fixed limits that only grow (with headroom) when the data outgrows them:
        # rescaling every frame would force a slow full redraw each time.
        self.ax_fd.set_xlim(0.0, 5.0)
        self.ax_ft.set_xlim(0.0, 5.0)
        for ax in (self.ax_fd, self.ax_ft):
            ax.set_ylim(-0.3, 8.0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color=GREY, alpha=0.15, lw=0.6)

    def start_sample(self, name, t_total=None):
        """Begin a new loading curve (one per probed material), lasting about t_total s."""
        self.name, self.color = name, SAMPLE_COLORS.get(name, NAVY)
        self.depths, self.forces, self.times = [], [], []
        self.line_fd, = self.ax_fd.plot([], [], color=self.color, lw=1.8, label=name)
        self.dot_fd, = self.ax_fd.plot([], [], "o", color=self.color, ms=7)
        self.line_ft, = self.ax_ft.plot([], [], color=self.color, lw=1.4, label=name)
        if t_total:
            self.ax_ft.set_xlim(0.0, max(self.ax_ft.get_xlim()[1], 1.02 * t_total))
        for ax in (self.ax_fd, self.ax_ft):
            ax.legend(fontsize=8, frameon=False, loc="upper left")
        self._set_blit_artists([self.line_fd, self.dot_fd, self.line_ft])
        self._redraw(full=True)

    def update_material(self, t, depth, f_act):
        self.f_peak = max(self.f_peak, f_act)
        self.n_samples += 1
        if self.n_samples % self.PLOT_EVERY == 0:
            self.times.append(t)
            self.depths.append(1e3 * depth)        # plotted in mm
            self.forces.append(f_act)
            self._redraw()

    def finish_sample(self, k_fit):
        """Overlay the least-squares fit F = k d on the current loading curve."""
        d_max = max(self.depths, default=0.0)
        self.ax_fd.plot([0.0, d_max], [0.0, k_fit * d_max * 1e-3], color=self.color, ls="--", lw=1.2)
        self.line_fd.set_label(f"{self.name}  (slope {k_fit:.0f} N/m)")
        self.ax_fd.legend(fontsize=8, frameon=False, loc="upper left")
        self._redraw(full=True)

    def add_result(self, name, K_hat, label):
        """List the stiffness estimate used for classification and the predicted class."""
        self.results.append(f"{name}: K̂ = {K_hat:4.0f} N/m -> {label}")
        self.results_text.set_text("\n".join(self.results))
        self._redraw(full=True)

    def _refresh_material(self):
        """Push the new data into the artists; True if the axes limits had to grow."""
        if self.name is None or not self.depths:
            return False
        self.line_fd.set_data(self.depths, self.forces)
        self.line_ft.set_data(self.times, self.forces)
        self.dot_fd.set_data([self.depths[-1]], [self.forces[-1]])
        grown = False
        d, f, t = self.depths[-1], self.f_peak, self.times[-1]
        if d > self.ax_fd.get_xlim()[1]:
            self.ax_fd.set_xlim(0.0, 1.3 * d)
            grown = True
        if t > self.ax_ft.get_xlim()[1]:
            self.ax_ft.set_xlim(0.0, 1.3 * t)
            grown = True
        if f > self.ax_fd.get_ylim()[1]:
            for ax in (self.ax_fd, self.ax_ft):
                ax.set_ylim(-0.3, 1.3 * f)
            grown = True
        return grown

    # ------------------------------------------------------------------- common
    def _set_blit_artists(self, artists):
        """Blitting: the static parts of the figure are rendered once (and again after any
        full redraw, e.g. a window resize) and cached; a refresh only redraws `artists`
        on top of that cache, which keeps the dashboard from slowing the simulation.
        Artists that stop changing (a finished material sample) become static again."""
        for a in self.blit_artists:
            a.set_animated(False)
        self.blit_artists = artists
        for a in artists:
            a.set_animated(True)

    def _on_draw(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self.blit_artists:
            self.fig.draw_artist(a)

    def _redraw(self, full=False):
        """Refresh the window (at most every REFRESH_PERIOD s, unless full=True). A full
        redraw re-renders the whole figure; it is needed after anything static changed."""
        now = time.perf_counter()
        if not full and now - self.last_update < self.REFRESH_PERIOD:
            return
        self.last_update = now
        if not plt.fignum_exists(self.fig.number):     # window closed by the user: keep simulating
            return
        grown = self._refresh_writing() if self.mode == "writing" else self._refresh_material()
        canvas = self.fig.canvas
        if full or grown or self.background is None or not self.blit_artists:
            canvas.draw()
        else:
            canvas.restore_region(self.background)
            for a in self.blit_artists:
                self.fig.draw_artist(a)
            canvas.blit(self.fig.bbox)
        canvas.flush_events()

    def close(self, wait=False):
        """Close the dashboard. With wait=True, first draw the final data and keep the
        window open (blocking) until the user closes it."""
        if wait and plt.fignum_exists(self.fig.number):
            self._set_blit_artists([])               # draw (and save) them like any artist
            self.fig.text(0.99, 0.01, "Finished - close this window to continue",
                          ha="right", va="bottom", fontsize=9, color=GREY)
            self._redraw(full=True)
            plt.ioff()
            plt.show()                       # blocks until the window is closed
        plt.ioff()
        plt.close(self.fig)
        plt.switch_backend("Agg")            # back to headless for the saved figures


# ==============================================================================
# WRITING & PROBING RUNNERS
# ==============================================================================

def _order_strokes(strokes, pen_xy):
    """Order and direction of one letter's strokes that minimise the total length of
    the pen-down connectors, starting from the pen at `pen_xy` (None: anywhere).
    Letters have at most 4 strokes, so every permutation and direction is tried;
    ties keep the font's own order and direction."""
    best_cost, best_seq = np.inf, strokes
    for perm in itertools.permutations(strokes):
        for flips in itertools.product((False, True), repeat=len(perm)):
            seq = [s[::-1] if flip else s for s, flip in zip(perm, flips)]
            cost, cur = 0.0, pen_xy
            for s in seq:
                if cur is not None:
                    cost += np.hypot(s[0][0] - cur[0], s[0][1] - cur[1])
                cur = s[-1]
            if cost < best_cost - 1e-6:            # 1 um: a mere rounding difference is a tie
                best_cost, best_seq = cost, seq
    return best_seq


def continuous_word_strokes(word):
    """The strokes of `word` joined into one pen-down polyline per space-separated
    word, so the pen never lifts between strokes or letters: each stroke ends with
    a short connector, drawn in contact, to the start of the next one (cursive-style)."""
    strokes = get_word_strokes(word)
    runs, run, i = [], [], 0
    for ch in word:
        n = len(LETTER_STROKES[ch if ch in LETTER_STROKES else ch.upper()])
        letter, i = strokes[i:i + n], i + n
        if not letter:                                 # space: lift the pen between words
            if run:
                runs.append(run)
            run = []
            continue
        for stroke in _order_strokes(letter, run[-1] if run else None):
            # skip the first point when the stroke starts where the pen already is
            same = run and np.allclose(run[-1], stroke[0], atol=1e-9)
            run.extend(stroke[1:] if same else stroke)
    if run:
        runs.append(run)
    return runs


def build_stroke_plan(world, letter_origin_xy, word=None, pattern=None, pen_lifts=False):
    """Plan of ("up" | "down", xy_from, xy_to) moves. By default the pen stays down
    for a whole word (see continuous_word_strokes); pen_lifts=True lifts it between
    every stroke of the font instead. The pen always lifts to move to the pattern."""
    strokes = []
    if word:
        strokes.extend(get_word_strokes(word) if pen_lifts else continuous_word_strokes(word))
    if pattern:
        pattern_strokes = generate_pattern(pattern)
        # With a word, draw the pattern 10 cm below it (the word occupies y in [0, 0.05]);
        # the pad is only +-0.2 m, so a sideways shift of 0.5 m would leave the pad.
        dy = -0.10 if word else 0.0
        pattern_strokes = [[(px, py + dy) for (px, py) in seg] for seg in pattern_strokes]
        strokes.extend(pattern_strokes)
        
    plan = []
    for stroke in strokes:
        pts = [np.array(letter_origin_xy) + np.array(pt) for pt in stroke]
        for i in range(len(pts)):
            if i == 0:
                if plan: plan.append(("up", plan[-1][2], pts[0]))
                else:    plan.append(("up", world.pre_contact_pos[:2], pts[0]))
            if i < len(pts) - 1 and np.linalg.norm(pts[i + 1] - pts[i]) > 1e-9:
                plan.append(("down", pts[i], pts[i+1]))
    return plan


TOUCH_DOWN_DURATION = 0.4      # s to lower the pen from PEN_LIFT_HEIGHT onto the surface (slow: no impact spike)


def stroke_duration(xy_from, xy_to):
    """Duration of one pen-down edge: STROKE_SPEED on average, but at least 0.15 s."""
    return max(np.linalg.norm(np.asarray(xy_to) - np.asarray(xy_from)) / STROKE_SPEED, 0.15)


def plan_duration(plan):
    """Simulated time run_writing_task() takes for `plan`, approach included."""
    total, prev_kind = APPROACH_DURATION, None
    for kind, xy_from, xy_to in plan:
        if kind == "up":
            total += PEN_UP_DURATION
        else:
            total += stroke_duration(xy_from, xy_to) + (TOUCH_DOWN_DURATION if prev_kind != "down" else 0.0)
        prev_kind = kind
    return total


def mean_squared_jerk(pos, mask, dt=DT):
    """Mean of |d^3 pos / dt^3|^2 (m^2/s^6) over the samples selected by `mask`.
    Jerk is differentiated inside each contiguous run of selected samples, so the
    jumps across pen lifts / touch-down do not show up as fake jerk spikes."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    squared = []
    for start, end in zip(edges[::2], edges[1::2]):
        seg = pos[start:end]
        if len(seg) < 8:
            continue
        jerk = seg
        for _ in range(3):
            jerk = np.gradient(jerk, dt, axis=0)
        squared.append(np.sum(jerk[3:-3] ** 2, axis=1))     # drop one-sided edge estimates
    return float(np.mean(np.concatenate(squared))) if squared else 0.0


def run_writing_task(world, ctrl, controller_mode, f_desired, gui=False, save_dir="outputs", tag="writing", word="Robot", pattern=None, live=False, pen_lifts=False):
    plan = build_stroke_plan(world, world.surface_center_xy, word=word, pattern=pattern, pen_lifts=pen_lifts)
    adm = AdmittanceController()

    # Ink: a mark on the surface under the tip wherever it presses on the page.
    ink = TrajectoryDrawer(world.cid, color=INK_COLOR, width=3.0, min_dist=0.0015) if gui else None
    title = f"{controller_mode.capitalize()} control: writing '{word}'" if word else f"{controller_mode.capitalize()} control"
    if pattern:
        title += f" + {pattern}" if word else f": {pattern}"
    dashboard = LiveDashboard(mode="writing", f_desired=f_desired, plan=plan,
                              t_total=plan_duration(plan), title=title) if live else None
    logger = TrialLogger()
    logger.start_timer()
    t = 0.0
    force = np.zeros(3)

    def advance(pos_des, gains, f_des, vel_des=None):
        """One control + physics step towards pos_des; then log, ink and plot the result."""
        nonlocal t, force
        tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"], vel_des=vel_des)
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()

        tool_pos, _, _, _ = world.arm.get_tool_pose()
        force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, world.static_ids, tool_pos)
        logger.record(t, tool_pos, pos_des, force=force, force_des=f_des, torque=torque)
        if ink is not None:
            if force[2] > INK_FORCE_THRESHOLD:
                ink.update((tool_pos[0], tool_pos[1], world.surface_top_z + INK_HEIGHT))
            else:
                ink.last_point = None             # pen off the page: break the ink line
        if dashboard is not None:
            dashboard.update_writing(t, tool_pos, pos_des, force, f_des)
        t += DT

    first_xy = plan[0][1]
    lift_pos = np.array([first_xy[0], first_xy[1], world.surface_top_z + PEN_LIFT_HEIGHT])
    traj_in = MinJerkTrajectory(world.pre_contact_pos, lift_pos, APPROACH_DURATION)
    for step in range(int(APPROACH_DURATION / DT)):
        pos_des, vel_des = traj_in.eval(step * DT)
        advance(pos_des, GAINS_PEN_UP, 0.0, vel_des)

    prev_kind = None
    t_down0 = 0.0
    for kind, xy_from, xy_to in plan:
        if kind == "up":
            p_from = np.array([xy_from[0], xy_from[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            p_to = np.array([xy_to[0], xy_to[1], world.surface_top_z + PEN_LIFT_HEIGHT])
            traj = MinJerkTrajectory(p_from, p_to, PEN_UP_DURATION)
            for step in range(int(PEN_UP_DURATION / DT)):
                pos_des, vel_des = traj.eval(step * DT)
                advance(pos_des, GAINS_PEN_UP, 0.0, vel_des)
            adm.reset()
        else:
            if prev_kind != "down":
                adm.reset()
                z_start = world.surface_top_z + PEN_LIFT_HEIGHT
                z_end = world.surface_top_z + TOOL_RADIUS
                traj_app = MinJerkTrajectory(np.array([xy_from[0], xy_from[1], z_start]), np.array([xy_from[0], xy_from[1], z_end]), TOUCH_DOWN_DURATION)
                for step in range(int(TOUCH_DOWN_DURATION / DT)):
                    pos_des, vel_des = traj_app.eval(step * DT)
                    advance(pos_des, GAINS_PEN_UP, 0.0, vel_des)
                t_down0 = t                       # time this stroke touched down

            duration = stroke_duration(xy_from, xy_to)
            xy_traj = MinJerkTrajectory(np.append(xy_from, 0), np.append(xy_to, 0), duration)
            for step in range(int(duration / DT)):
                xy_pos, xy_vel = xy_traj.eval(step * DT)
                f_meas_z = force[2]               # measured after the previous physics step

                if controller_mode == "admittance":
                    xc = adm.step(f_meas_z, f_desired, DT)
                    z_des = world.surface_top_z + TOOL_RADIUS - ADMITTANCE_NOMINAL_PENETRATION + xc
                    z_vel = adm.xc_dot                # the stiff inner loop tracks xc, motion included
                    gains = GAINS_INNER_STIFF
                else:
                    # Impedance writing: no force feedback. The soft normal-axis spring
                    # is given a setpoint *below* the surface so that spring (in series
                    # with the surface) pushes with f_desired:
                    #   f = k_z k_c / (k_z + k_c) * penetration
                    #   => penetration = f_desired * (1/k_z + 1/k_c)
                    # The penetration is ramped in so touch-down does not hit as a step.
                    # Along the page the spring is stiffer (anisotropic impedance), so
                    # friction (~mu * f_desired) cannot pull the tip off the letters.
                    gains = dict(GAINS_IMPEDANCE, K_trans=np.array(
                        [IMPEDANCE_K_TANGENT, IMPEDANCE_K_TANGENT, GAINS_IMPEDANCE["K_trans"][2]]))
                    k_z = gains["K_trans"][2]
                    penetration = f_desired * (1.0 / k_z + 1.0 / CONTACT_STIFFNESS_EST)
                    s = np.clip((t - t_down0) / IMPEDANCE_RAMP_TIME, 0.0, 1.0)
                    penetration *= 10 * s**3 - 15 * s**4 + 6 * s**5      # min-jerk ramp 0 -> 1
                    z_des = world.surface_top_z + TOOL_RADIUS - penetration
                    z_vel = 0.0                       # virtual setpoint inside the surface: the
                                                      # tip never gets there, so no feedforward

                advance(np.array([xy_pos[0], xy_pos[1], z_des]), gains, f_desired,
                        np.array([xy_vel[0], xy_vel[1], z_vel]))
        prev_kind = kind
    
    logger.stop_timer()
    if dashboard: dashboard.close(wait=True)

    os.makedirs(save_dir, exist_ok=True)
    arr = logger.arrays()
    
    # Metrics over the pen-down samples only (force_des > 0): RMSE of Fz against the
    # desired force, and the mean squared jerk of the tool path.
    pen_down = arr["force_des"] > 0
    f_err = arr["force"][pen_down, 2] - arr["force_des"][pen_down]
    rmse = float(np.sqrt(np.mean(f_err ** 2))) if f_err.size else 0.0
    smoothness = mean_squared_jerk(arr["pos"], pen_down)
    
    plot_force_tracking(arr, os.path.join(save_dir, f"{tag}_force_tracking.png"), title=f"{tag}: Force Tracking While Writing")
    plot_xy_path(arr, os.path.join(save_dir, f"{tag}_xy_path.png"), title=f"{tag}: Traced Path (top-down)")
    
    summary = dict(force_tracking_rmse_N=rmse, path_smoothness_jerk2=smoothness, wall_clock_time_s=logger.wall_clock_time(), sim_time_s=float(arr["t"][-1]))
    save_json(os.path.join(save_dir, f"{tag}_summary.json"), summary)
    print(f"[{tag}] force RMSE = {rmse:.3f} N")
    return summary, arr


def probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, tag, fixed_penetration=0.010, settle_duration=3.0, dashboard=None):
    gains = dict(K_trans=np.array([600., 600., 600.]), K_rot=np.array([30., 30., 30.]), damping_ratio=1.8)
    just_touching_z = pad_top_z + TOOL_RADIUS
    target = np.array([xy[0], xy[1], just_touching_z - fixed_penetration])

    hover = np.array([xy[0], xy[1], just_touching_z + 0.05])
    q_hover = world.arm.inverse_kinematics(hover, world.nominal_quat)
    world.arm.set_joint_positions(q_hover)
    world.arm.sync_tool_pose()

    traj = MinJerkTrajectory(hover, target, APPROACH_DURATION)
    if dashboard:
        dashboard.start_sample(tag, t_total=APPROACH_DURATION + settle_duration)
    logger = TrialLogger()
    t = 0.0
    for step in range(int((APPROACH_DURATION + settle_duration) / DT)):
        pos_des, _ = traj.eval(t)
        tau, _ = ctrl.step(pos_des, world.nominal_quat, gains["K_trans"], gains["K_rot"], gains["damping_ratio"])
        world.arm.apply_joint_torques(tau)
        p.stepSimulation(physicsClientId=world.cid)
        world.arm.clamp_joint_velocities()

        tool_pos, _, _, _ = world.arm.get_tool_pose()
        force, torque, _ = get_contact_wrench(world.cid, world.arm.tool_id, [pad_id], tool_pos)
        
        depth = max(0.0, just_touching_z - tool_pos[2])     # measured penetration of the tip
        logger.record(t, tool_pos, pos_des, force=force, force_des=0.0, torque=torque)
        if dashboard:
            dashboard.update_material(t, depth, force[2])
        t += DT

    arr = logger.arrays()
    forces = arr["force"][:, 2]
    depths = np.maximum(0.0, just_touching_z - arr["pos_des"][:, 2])

    # Least-squares slope of F = K * depth through the origin over the samples in
    # contact:  K_hat = argmin sum (F - K d)^2 = (d . F) / (d . d).
    # `depths` is the *commanded* penetration, so K_hat is the stiffness of the
    # whole chain (controller spring in series with the material), not the material alone.
    in_contact = depths > 1e-4
    d, f = depths[in_contact], forces[in_contact]
    K_hat = float(d @ f / (d @ d)) if d.size else 0.0

    if dashboard:
        # The live curve is force vs. the *measured* penetration, so its own slope is
        # the contact stiffness the tip actually feels (no controller spring in series).
        d_meas = np.maximum(0.0, just_touching_z - arr["pos"][:, 2])
        touching = d_meas > 1e-4
        dm, fm = d_meas[touching], forces[touching]
        dashboard.finish_sample(float(dm @ fm / (dm @ dm)) if dm.size else 0.0)

    os.makedirs(save_dir, exist_ok=True)
    steady_force = float(np.percentile(forces[-max(int(2.0/DT), 5):], 75))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(arr["t"], forces, color=RED, lw=1.8)
    ax.axhline(steady_force, color=GREY, ls="--", lw=1.2, label=f"steady state: F={steady_force:.1f} N, K̂={K_hat:.0f} N/m")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Contact force (N)")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Material Probe -- {tag}", fontsize=12, weight="bold", color=NAVY)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tag}_loading_curve.png"), dpi=180, facecolor="white")
    plt.close(fig)
    return K_hat


# ==============================================================================
# MAIN EXECUTIONS
# ==============================================================================

def run_part1(save_dir="outputs/part1"):
    print("--- Part 1: Trajectory Generation ---")
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    shapes = ["circle", "spiral", "infinity", "heart"]
    for i, shape in enumerate(shapes):
        strokes = generate_pattern(shape, scale=0.1)
        ax = axes[i]
        for s in strokes:
            pts = np.array(s)
            ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=2)
        ax.set_aspect("equal")
        ax.set_title(shape.capitalize())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shapes.png"), dpi=150)
    plt.close()

def aim_gui_camera(world, gui):
    """Point the PyBullet viewer at the writing pad (default camera looks elsewhere)."""
    if gui:
        # yaw 0: world +x is screen-right and +y is screen-up, so text reads left to right
        p.resetDebugVisualizerCamera(cameraDistance=0.75, cameraYaw=0, cameraPitch=-55,
                                     cameraTargetPosition=[*world.surface_center_xy, world.surface_top_z],
                                     physicsClientId=world.cid)


def hold_gui(gui):
    """Keep the PyBullet window open after the run (it would vanish on disconnect)."""
    if gui:
        try:
            input("Simulation finished -- press Enter to close the PyBullet window... ")
        except EOFError:
            time.sleep(5)


def run_part2_writing(gui=False, save_dir="outputs/part3", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False, pen_lifts=False):
    print("--- Part 2 (continued): Impedance Writing ---")
    world = build_world(gui=gui)
    aim_gui_camera(world, gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "impedance", f_desired, gui=gui, save_dir=save_dir, tag="part3_impedance", word=word, pattern=pattern, live=live, pen_lifts=pen_lifts)
    hold_gui(gui)
    world.disconnect()
    return s

def run_part2_writing_admittance(gui=False, save_dir="outputs/part4", f_desired=F_DESIRED, word="Robot", pattern="circle", live=False, pen_lifts=False):
    print("--- Part 3: Admittance Writing ---")
    world = build_world(gui=gui)
    aim_gui_camera(world, gui)
    ctrl = CartesianImpedanceController(world.arm)
    s, _ = run_writing_task(world, ctrl, "admittance", f_desired, gui=gui, save_dir=save_dir, tag="part4_admittance", word=word, pattern=pattern, live=live, pen_lifts=pen_lifts)
    hold_gui(gui)
    world.disconnect()
    return s

def run_part3(save_dir="outputs/part5", gui=False, word="Robot", pattern="circle", live=False, pen_lifts=False):
    print("--- Part 4: Analysis and Comparison ---")
    os.makedirs(save_dir, exist_ok=True)
    s3 = run_part2_writing(gui=gui, word=word, pattern=pattern, live=live, pen_lifts=pen_lifts)
    s4 = run_part2_writing_admittance(gui=gui, word=word, pattern=pattern, live=live, pen_lifts=pen_lifts)
    
    dynamic_cmp = {"Impedance (Part 2)": s3, "Admittance (Part 3)": s4}
    bar_compare(["Force RMSE (N)"], {k: [v["force_tracking_rmse_N"]] for k, v in dynamic_cmp.items()}, "RMSE (N)", os.path.join(save_dir, "part5_writing_force_rmse.png"), "Writing: Force Tracking RMSE")
    bar_compare(["Path Smoothness (mean sq. jerk)"], {k: [v["path_smoothness_jerk2"]] for k, v in dynamic_cmp.items()}, "Mean squared jerk", os.path.join(save_dir, "part5_writing_smoothness.png"), "Writing: Motion Smoothness")

def run_part4(gui=False, save_dir="outputs/part6", live=False):
    print("--- Part 5: Haptic Material Identification ---")
    os.makedirs(save_dir, exist_ok=True)
    world = build_world(gui=gui, with_material_samples=True)
    aim_gui_camera(world, gui)
    ctrl = CartesianImpedanceController(world.arm)
    dashboard = LiveDashboard(mode="material") if live else None
    results = {}
    for name, (pad_id, pad_top_z, true_label) in world.material_ids.items():
        spec = MATERIAL_SAMPLES[name]
        xy = world.surface_center_xy + np.array(spec["offset_xy"])
        K_hat = probe_material(world, ctrl, pad_id, pad_top_z, xy, save_dir, name, dashboard=dashboard)

        # soft: K_hat < t0 <= medium < t1 <= hard
        t0, t1 = MATERIAL_CLASS_THRESHOLDS
        pred_label = "soft" if K_hat < t0 else ("medium" if K_hat < t1 else "hard")
        if dashboard:
            dashboard.add_result(name, K_hat, pred_label)

        results[name] = dict(true_stiffness_N_per_m=spec["contact_stiffness"], estimated_stiffness_N_per_m=K_hat, true_label=true_label, predicted_label=pred_label, correct=(pred_label == true_label))
        print(f"[Part 5] {name}: estimated K̂={K_hat:.0f} N/m -> predicted={pred_label}")
    n_correct = sum(1 for r in results.values() if r["correct"])
    save_json(os.path.join(save_dir, "part6_results.json"), dict(accuracy=n_correct/3.0, results=results))
    if dashboard:
        dashboard.close(wait=True)
    hold_gui(gui)
    world.disconnect()
    return results

def main():
    parser = argparse.ArgumentParser(description="Force-Controlled Writing")
    parser.add_argument("--part", choices=["1", "2", "3", "4", "5", "all"], default="all")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--word", type=str, default="Ravi", help="Word to write")
    parser.add_argument("--pattern", type=str, default="circle", help="Pattern to draw")
    parser.add_argument("--live", action="store_true", help="Enable live dashboard visualization")
    parser.add_argument("--pen-lifts", action="store_true",
                        help="Lift the pen between every stroke (default: pen stays down for a whole word)")
    args = parser.parse_args()
    
    pattern = None if args.pattern == "none" else args.pattern
    
    # 1 = shapes, 2 = impedance writing, 3 = admittance writing, 4 = comparison, 5 = material ID
    if args.part in ["1", "all"]: run_part1()
    if args.part in ["2", "all"]: run_part2_writing(gui=args.gui, word=args.word, pattern=pattern, live=args.live, pen_lifts=args.pen_lifts)
    if args.part in ["3", "all"]: run_part2_writing_admittance(gui=args.gui, word=args.word, pattern=pattern, live=args.live, pen_lifts=args.pen_lifts)
    if args.part in ["4", "all"]: run_part3(gui=args.gui, word=args.word, pattern=pattern, live=args.live, pen_lifts=args.pen_lifts)
    if args.part in ["5", "all"]: run_part4(gui=args.gui, live=args.live)

if __name__ == '__main__':
    main()

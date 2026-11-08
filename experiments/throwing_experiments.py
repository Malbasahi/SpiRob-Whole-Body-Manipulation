#!/usr/bin/env python3
# test_three_roll_multi_episode_throw_FIXED_RELEASE.py
#
# What this script does (exactly per your request):
#   - OBJECT (O1) position is FIXED every episode (same coordinate).
#   - THREE rolls always exist:
#       Roll1 BACK  (complete)
#       Roll2 FRONT (complete)  <-- lengthened
#       Roll3 VARIES per episode (direction + duration/velocity + hold/release style)
#   - MULTI-EPISODE (4+): only the LAST roll behavior changes.
#   - Some episodes HOLD (no release).
#   - Some episodes THROW: hold for a while, then RELEASE (actually releases).
#       -> Uses TWO robust release mechanisms:
#            (1) temporary friction drop on O1 during release window
#            (2) explicit UNWIND roll to peel the spiral off the object
#
# Output:
#   SAVE_DIR/
#     best_coeffs.npy
#     episode_ep1_RIGHT_ctrl.npy
#     episode_ep2_RIGHT_ctrl.npy
#     ...
#     multi_episode_summary.json
#
# Requirements:
#   pip install mujoco cma numpy
#
# Notes:
#   - This targets your XML actuator style (muscle-like ctrl [0..1]).
#   - If you still see "stuck" objects: decrease release_friction_scale (e.g., 0.02),
#     increase unwind_sec (e.g., 0.8), and/or increase UNWRAP_SEC.

import os, time, math, json
import numpy as np

# ---------------- Determinism-ish thread caps ----------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
from cma import CMAEvolutionStrategy

try:
    import mujoco.viewer as mjv
    HAVE_MJV = True
except Exception:
    HAVE_MJV = False


# ========================= USER CONFIG =========================
XML_PATH   = str(Path(__file__).resolve().parents[1] / "SpiRob.xml")
SAVE_DIR   = str(Path(__file__).resolve().parents[1] / "outputs" / "throwing")                                           
os.makedirs(SAVE_DIR, exist_ok=True)

EE_SITE_NAME   = "s23_3"
OBJ_BODY_NAME  = "O1"
OBJ_JOINT_NAME = "O1_freejoint"
KEY_RIGHT_NAME = "right_start"

# ---------------- SIM / ACTUATION ----------------
DT_SIM            = 0.002
CTRL_HZ           = 220.0
PLAYBACK_SLOWDOWN = 2.0
POST_VIEW_SEC     = 0.4

INIT_PRESSURE     = 0.10
PRESSURE_TAU      = 0.10
PRESSURE_SLEW     = 1.60
SMOOTH_ALPHA      = 0.16

# ---------------- GEOMETRY (metrics) ----------------
CYL_R            = 0.015
HOLD_TARGET_R    = CYL_R + 0.014
NEAR_R           = CYL_R + 0.026

COVER_BAND_IN    = CYL_R - 0.002
COVER_BAND_OUT   = CYL_R + 0.009
COVER_MIN_COUNT  = 9

# ---------------- PHASE TIMING ----------------
T_APPROACH     = 1.20

BACKROLL_SEC   = 0.85          # Roll1 BACK (complete)
FRONTROLL_SEC  = 2.20          # Roll2 FRONT (complete, longer)
ROLL_DWELL_SEC = 0.12

GRASP_RAMP_SEC       = 0.22
GRASP_SUPPORT_START  = 0.60
GRASP_SUPPORT_FINAL  = 0.92
GRASP_ROLLER_START   = 0.98
GRASP_ROLLER_FINAL   = 1.00
GRASP_RAMP_STEPS     = max(1, int(GRASP_RAMP_SEC / DT_SIM))

ROLLER_IDX_BACK  = 0
ROLLER_IDX_FRONT = 1

# After release: brief "shake" to break static capture (low-ish controls)
UNWRAP_SEC          = 0.35
POST_RELEASE_TAIL_S = 0.25

# A base third-roll (complete) duration; episodes override this to change velocity
BASE_THIRDROLL_SEC  = 2.10

# ---------------- CMA WAVEFORM ----------------
N_HARM = 2


# ========================= EPISODE TARGETS =========================
# Fixed object pose. CMA changes only throw-generation knobs, not object location.
# desired_dir_xz is the target throw direction in the simulation x-z plane.
# desired_speed is the target object speed at release, in m/s.
#
# The first two rolls and grasp ramp stay identical across episodes.
# Per episode, CMA optimizes: third-roll direction, third-roll duration/speed,
# hold tail, release ramp, friction scale, unwind duration, unwrap strength.
EPISODE_TARGETS = [
    dict(name="throw_pos_x_slow",   desired_dir_xz=(+1.0, 0.0), desired_speed=0.18),
    dict(name="throw_pos_x_fast",   desired_dir_xz=(+1.0, 0.0), desired_speed=0.32),
    dict(name="throw_up_mid",      desired_dir_xz=( 0.0,+1.0), desired_speed=0.24),
    dict(name="throw_diag_fast",   desired_dir_xz=(+1.0,+0.7), desired_speed=0.34),
    dict(name="throw_neg_x_mid",   desired_dir_xz=(-1.0, 0.0), desired_speed=0.24),
]

# Bounds for CMA-decoded throw plan. Keep these conservative first; widen only
# after confirming the object reliably releases.
THROW_PARAM_BOUNDS = dict(
    thirdroll_sec=(0.90, 3.20),
    extra_hold_tail=(0.00, 0.75),
    release_after=(0.00, 0.25),
    release_ramp=(0.08, 0.45),
    release_friction_scale=(0.015, 0.12),
    unwind_sec=(0.20, 1.10),
    unwrap_strength=(0.10, 0.38),
)

# Throw objective weights. Direction is intentionally dominant; speed is second.
W_DIR   = 8.0
W_SPEED = 5.0
W_GRASP = 1.0
W_RELEASE = 2.0
W_TIME  = 0.05

# ========================= Helpers =========================
def site_id(model, name, default_last=False):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0 and default_last:
        return model.nsite - 1
    if sid < 0:
        raise RuntimeError(f"Site not found: {name}")
    return sid

def reset_to_key(model, data, key_name):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if kid < 0:
        raise RuntimeError(f"Keyframe not found: {key_name}")
    mujoco.mj_resetDataKeyframe(model, data, kid)

def cylinder_face_on_quat_wxyz():
    # face-on about X (same as your prior)
    s = math.sin(math.pi/4)
    c = math.cos(math.pi/4)
    return (c, s, 0.0, 0.0)

def set_object_pose_freejoint(model, data, xyz, quat_wxyz):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {OBJ_JOINT_NAME}")
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr+3]   = np.asarray(xyz, float)
    data.qpos[adr+3:adr+7] = np.asarray(quat_wxyz, float)
    mujoco.mj_forward(model, data)

def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")
    return bid

def get_object_center_from_state(model, data):
    bid = body_id(model, OBJ_BODY_NAME)
    mujoco.mj_forward(model, data)
    return data.xipos[bid].copy()

def object_freejoint_qvel_xz(model, data):
    """Return object translational velocity in the x-z plane from O1_freejoint."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJ_JOINT_NAME)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {OBJ_JOINT_NAME}")
    adr = model.jnt_dofadr[jid]
    v = np.asarray(data.qvel[adr:adr+3], float)
    return np.array([v[0], v[2]], float)

def normalize_xz(vec, fallback=(1.0, 0.0)):
    v = np.asarray(vec, float).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v = np.asarray(fallback, float).reshape(2)
        n = float(np.linalg.norm(v))
    return v / max(n, 1e-9)

def polar_of(vec, center):
    dx, _, dz = vec - center
    return math.hypot(dx, dz), math.atan2(dz, dx)

def subchain_ids(model, start_name="B12", end_name="B23"):
    ids = []
    si = int(start_name[1:])
    ei = int(end_name[1:])
    for k in range(si, ei+1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"B{k}")
        if bid >= 0:
            ids.append(bid)
    return ids

def unwrap_angle_seq(a):
    return np.unwrap(np.asarray(a, float))

def coverage_score(data, body_ids, goal_xyz):
    gx, gz = goal_xyz[0], goal_xyz[2]
    th, rr = [], []
    for bid in body_ids:
        x, _, z = data.xipos[bid]
        dx, dz = x - gx, z - gz
        th.append(math.atan2(dz, dx))
        rr.append(math.hypot(dx, dz))
    thetas = np.asarray(th)
    r = np.asarray(rr)
    near = (r >= max(COVER_BAND_IN, 1e-6)) & (r <= COVER_BAND_OUT)
    cnt = int(np.sum(near))
    if cnt > 1:
        ang = np.sort((thetas[near] + 2*math.pi) % (2*math.pi))
        diffs = np.diff(np.concatenate([ang, ang[:1] + 2*math.pi]))
        span = 2*math.pi - np.max(diffs)
    else:
        span = 0.0
    span_deg = span * 180.0 / math.pi
    score = 0.0
    score += min(cnt/max(COVER_MIN_COUNT, 1), 1.5) * 0.5
    score += min(span_deg/360.0, 1.5) * 0.4
    return float(score)

def area_fill_score(data, body_ids, goal_xyz, rays=24):
    gx, gz = goal_xyz[0], goal_xyz[2]
    pts = np.array([[data.xipos[bid,0], data.xipos[bid,2]] for bid in body_ids], float)
    vecs = pts - np.array([gx, gz])
    ang  = (np.arctan2(vecs[:,1], vecs[:,0]) + 2*np.pi) % (2*np.pi)
    rad  = np.linalg.norm(vecs, axis=1)
    edges = np.linspace(0, 2*np.pi, rays+1)
    rmin  = np.full(rays, np.inf)
    for a, r in zip(ang, rad):
        k = min(rays-1, np.searchsorted(edges, a, side='right') - 1)
        rmin[k] = min(rmin[k], r)
    band_out = COVER_BAND_OUT
    rmin = np.where(np.isfinite(rmin), rmin, band_out)
    rclamp = np.clip(rmin, CYL_R*0.8, CYL_R*1.6)
    dθ = 2*np.pi / rays
    area = 0.5 * np.sum(rclamp**2) * dθ
    ideal_r = np.clip(CYL_R + 0.004, CYL_R*0.8, CYL_R*1.6)
    ideal_area = 0.5 * (ideal_r**2) * (2*np.pi)
    fill01 = float(np.clip(area / max(ideal_area, 1e-9), 0.0, 1.5))
    return float(area), float(fill01)


# ---------------- Viewer wrapper ----------------
class SafeViewer:
    def __init__(self, model, data, show=True):
        self.kind = None
        self.ctx  = None
        self.v    = None
        if not show:
            return
        if HAVE_MJV:
            try:
                self.ctx = mjv.launch_passive(model, data)
                self.v = self.ctx.__enter__()
                self.kind = "mjv"
                return
            except Exception:
                pass
        try:
            import mujoco_viewer as mv
            self.v = mv.MujocoViewer(model, data)
            self.kind = "mv"
        except Exception:
            self.v = None

    def draw(self):
        if self.v is None:
            return
        try:
            self.v.sync() if self.kind == "mjv" else self.v.render()
        except Exception:
            pass

    def close(self):
        if self.v is None:
            return
        try:
            self.ctx.__exit__(None, None, None) if self.kind == "mjv" else self.v.close()
        except Exception:
            pass


# ---------------- Pressure dynamics ----------------
def apply_pressure_step(d, u01, p, p_smooth, dt):
    alpha   = dt / max(PRESSURE_TAU, 1e-6)
    desired = p + alpha*(u01 - p)
    max_step = PRESSURE_SLEW * dt
    p = np.clip(p + np.clip(desired - p, -max_step, max_step), 0.0, 1.0)
    p_smooth = (1.0 - SMOOTH_ALPHA) * p_smooth + SMOOTH_ALPHA * p
    d.ctrl[:] = p_smooth
    return p, p_smooth


# ========================= Friction override (for REAL release) =========================
def geom_ids_for_body(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {body_name}")
    return np.where(model.geom_bodyid == bid)[0].astype(int).tolist()

def snapshot_geom_friction(model, geom_ids):
    return {gid: model.geom_friction[gid].copy() for gid in geom_ids}

def restore_geom_friction(model, backup):
    for gid, fr in backup.items():
        model.geom_friction[gid] = fr.copy()

def set_geom_friction_scale_from_backup(model, geom_ids, backup, scale):
    scale = float(scale)
    for gid in geom_ids:
        model.geom_friction[gid] = backup[gid] * scale


# ========================= CMA waveform =========================
def pack_coeffs(b0, bc, bs, s0, sc, ss, phase, kappa_raw, eps_raw, warp_k, warp_c,
                s1_raw, s2_raw, s3_raw, s4_raw, w_raw,
                aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw):
    return np.concatenate([
        np.array([b0], float), bc, bs,
        np.array([s0], float), sc, ss,
        np.array([phase, kappa_raw, eps_raw, warp_k, warp_c,
                  s1_raw, s2_raw, s3_raw, s4_raw, w_raw,
                  aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw], float)
    ])

def squash_params(x):
    idx = 0
    b0 = x[idx]; idx += 1
    bc = x[idx:idx+N_HARM]; idx += N_HARM
    bs = x[idx:idx+N_HARM]; idx += N_HARM
    s0 = x[idx]; idx += 1
    sc = x[idx:idx+N_HARM]; idx += N_HARM
    ss = x[idx:idx+N_HARM]; idx += N_HARM
    phase = x[idx]; idx += 1

    _kappa   = 0.55 + 0.15*np.tanh(x[idx]); idx += 1
    _epsilon = 0.04 * np.tanh(x[idx]);      idx += 1

    warp_k = float(np.clip(x[idx], -2.5, 2.5)); idx += 1
    warp_c = float(np.clip(x[idx],  0.05, 0.95)); idx += 1

    s1 = 0.15 + 0.30 * (1/(1+np.exp(-x[idx]))); idx += 1
    s2 = s1 + (0.08 + 0.18*(1/(1+np.exp(-x[idx])))); idx += 1
    s3 = s2 + (0.12 + 0.22*(1/(1+np.exp(-x[idx])))); idx += 1
    s4 = s3 + (0.06 + 0.16*(1/(1+np.exp(-x[idx])))); idx += 1
    s4 = min(s4, 0.97)
    w  = 0.02 + 0.08 * (1/(1+np.exp(-x[idx]))); idx += 1

    aF1 = 0.45 + 0.70 * (1/(1+np.exp(-x[idx]))); idx += 1
    aB1 = 0.35 + 0.70 * (1/(1+np.exp(-x[idx]))); idx += 1
    aF2 = 0.50 + 0.70 * (1/(1+np.exp(-x[idx]))); idx += 1
    aB2 = 0.35 + 0.70 * (1/(1+np.exp(-x[idx]))); idx += 1
    aF3 = 0.55 + 0.70 * (1/(1+np.exp(-x[idx]))); idx += 1

    # enforce symmetry (no net roll bias in approach)
    kappa   = 0.50
    epsilon = 0.0

    return (b0, bc, bs, s0, sc, ss, phase, kappa, epsilon, warp_k, warp_c,
            s1, s2, s3, s4, w, aF1, aB1, aF2, aB2, aF3)

def build_time_warp(N, warp_k, warp_c):
    tau_raw = np.linspace(0.0, 1.0, N, endpoint=True)
    beta = 6.0
    speed = np.exp(warp_k * np.tanh(beta*(tau_raw - warp_c)))
    speed = np.clip(speed, 1e-3, None)
    s = np.cumsum(speed)
    return (s - s[0]) / (s[-1] - s[0] + 1e-12)

def _stretch_phase1(s1, s2, s3, s4, extra_frac=0.20, min_win=0.04):
    l1 = max(s1 - 0.0, 1e-6)
    l2 = max(s2 - s1, 1e-6)
    l3 = max(s3 - s2, 1e-6)
    l4 = max(s4 - s3, 1e-6)
    l5 = max(1.0 - s4, 1e-6)
    total = l1+l2+l3+l4+l5
    borrowable = max((l2-min_win),0.0) + max((l3-min_win),0.0) + max((l4-min_win),0.0) + max((l5-min_win),0.0)
    extra = min(extra_frac*total, borrowable)
    if extra <= 1e-9:
        return float(s1), float(s2), float(s3), float(s4)
    tail = l2+l3+l4+l5
    def red(li): return extra*(li/max(tail,1e-12))
    l1n = l1 + extra
    l2n = max(l2 - red(l2), min_win)
    l3n = max(l3 - red(l3), min_win)
    l4n = max(l4 - red(l4), min_win)
    l5n = max(l5 - red(l5), min_win)
    scale = total / (l1n+l2n+l3n+l4n+l5n)
    l1n*=scale; l2n*=scale; l3n*=scale; l4n*=scale; l5n*=scale
    s1n = l1n
    s2n = s1n + l2n
    s3n = s2n + l3n
    s4n = s3n + l4n
    return float(s1n), float(s2n), float(s3n), float(s4n)

def smooth_steps(tau, s1, s2, s3, s4, w):
    def S(x): return 0.5*(1+np.tanh(x/(w+1e-6)))
    sig1 = 1 - S(tau - s1)
    sig2 = S(tau - s1) * (1 - S(tau - s2))
    sig3 = S(tau - s2) * (1 - S(tau - s3))
    sig4 = S(tau - s3) * (1 - S(tau - s4))
    sig5 = S(tau - s4)
    Z = np.clip(sig1+sig2+sig3+sig4+sig5, 1e-6, None)
    return sig1/Z, sig2/Z, sig3/Z, sig4/Z, sig5/Z

def control_trajectory_from_coeffs(T, ctrl_hz, coeffs, dir_sign=+1.0):
    (b0, bc, bs, s0, sc, ss, phase, _, _, warp_k, warp_c,
     s1, s2, s3, s4, w, aF1, aB1, aF2, aB2, aF3) = squash_params(coeffs)

    s1, s2, s3, s4 = _stretch_phase1(s1, s2, s3, s4)
    N = int(T * ctrl_hz)
    tau = build_time_warp(N, warp_k, warp_c)

    b = b0 * np.ones_like(tau)
    for i in range(1, N_HARM+1):
        b += bc[i-1]*np.cos(2*math.pi*i*tau) + bs[i-1]*np.sin(2*math.pi*i*tau)
    base = 1.0/(1.0 + np.exp(-b))

    s = s0 * np.ones_like(tau)
    for i in range(1, N_HARM+1):
        ang = 2*math.pi*i*tau + phase
        s += sc[i-1]*np.cos(ang) + ss[i-1]*np.sin(ang)

    sig1, sig2, sig3, sig4, sig5 = smooth_steps(tau, s1, s2, s3, s4, w)
    sign = +1.0 if dir_sign >= 0 else -1.0
    env = sign*(+aF1*sig1 - aB1*sig2 + aF2*sig3 - aB2*sig4 + aF3*sig5)

    uL = base + 0.50*env
    uR = base - 0.50*env
    u  = np.stack([uL, uR], axis=1)
    return np.clip(u, 0.0, 1.0)


# ========================= Roll / grasp profiles =========================
def fast_roll_profile(progress01, roller_idx, keep_other_low):
    """Roll profile:
       - roller ~ [0.97..1.0]
       - other:
         * keep_other_low True  -> ~0.15 (pure roll / peel)
         * keep_other_low False -> ramps to ~0.84 (catch/hold bias)
    """
    p = float(np.clip(progress01, 0.0, 1.0))
    roller = 0.97 + 0.03*(p**2)
    other_base = 0.15

    if keep_other_low:
        other = other_base
    else:
        if p < 0.25:
            other = other_base
        elif p < 0.75:
            a = (p - 0.25) / 0.50
            a = np.clip(a, 0.0, 1.0)
            a = a*a*(3 - 2*a)
            other = other_base + a*(0.84 - other_base)
        else:
            other = 0.84

    if roller_idx == 0:
        u = np.array([roller, other], float)
    else:
        u = np.array([other, roller], float)
    return np.clip(u, 0.0, 1.0)

def perfect_grasp_control(roller_idx, step_idx):
    t = float(np.clip(step_idx / max(GRASP_RAMP_STEPS, 1), 0.0, 1.0))
    s = t*t*(3 - 2*t)
    support = GRASP_SUPPORT_START + s*(GRASP_SUPPORT_FINAL - GRASP_SUPPORT_START)
    roller  = GRASP_ROLLER_START  + s*(GRASP_ROLLER_FINAL  - GRASP_ROLLER_START)
    if roller_idx == 0:
        u = np.array([roller, support], float)
    else:
        u = np.array([support, roller], float)
    return np.clip(u, 0.0, 1.0)

def lerp_smooth(u0, u1, frac01):
    frac01 = float(np.clip(frac01, 0.0, 1.0))
    s = frac01*frac01*(3 - 2*frac01)
    return (1.0 - s)*u0 + s*u1

def unwrap_command(t, t0, t1, direction, strength):
    """Low-ish shake in chosen direction to help detach."""
    strength = float(np.clip(strength, 0.0, 0.40))
    if t1 <= t0:
        return np.array([0.0, 0.0], float)
    phase = 2*math.pi * 6.0 * ((t - t0) / (t1 - t0))
    osc = 0.5 + 0.5*math.sin(phase)
    hi = np.clip(0.10 + strength*osc, 0.0, 1.0)
    lo = 0.05
    if direction == "front":
        # bias towards FRONT roller idx
        return np.array([lo, hi], float)
    else:
        return np.array([hi, lo], float)


# ========================= Cost & CMA optimize once =========================
def default_coeffs():
    b0  = 0.40
    bc  = np.array([+0.08, +0.00])
    bs  = np.array([+0.00, +0.00])
    s0  = 0.12
    sc  = np.array([+0.14, -0.04])
    ss  = np.array([+0.00, +0.00])
    phi = 0.0
    kappa_raw = 0.0; eps_raw = 0.0
    warp_k = -0.8; warp_c = 0.60
    s1_raw, s2_raw, s3_raw, s4_raw, w_raw = -0.3, +0.3, +0.7, +1.1, -0.2
    aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw = 0.0, 0.4, 0.2, 0.5, 0.3
    return pack_coeffs(
        b0, bc, bs, s0, sc, ss, phi,
        kappa_raw, eps_raw, warp_k, warp_c,
        s1_raw, s2_raw, s3_raw, s4_raw, w_raw,
        aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw
    )

def cost_right_only(model, ee_sid, goal_xyz_fixed, coeffs):
    d = mujoco.MjData(model)
    reset_to_key(model, d, KEY_RIGHT_NAME)
    set_object_pose_freejoint(model, d, goal_xyz_fixed, cylinder_face_on_quat_wxyz())

    try:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance  = min(model.opt.tolerance, 1e-9)

    # Optimize only approach + roll1 + roll2 + grasp success (ignore roll3)
    T_OPT = T_APPROACH + BACKROLL_SEC + FRONTROLL_SEC + ROLL_DWELL_SEC + GRASP_RAMP_SEC + 0.40
    seq_u = control_trajectory_from_coeffs(T_OPT, CTRL_HZ, coeffs, dir_sign=+1.0)
    ctrl_dt = 1.0/CTRL_HZ
    steps_per = max(1, int(ctrl_dt/DT_SIM))

    nu = model.nu
    p = np.full(nu, INIT_PRESSURE, float)
    p_smooth = p.copy()

    mujoco.mj_forward(model, d)
    tip = d.site_xpos[ee_sid].copy()
    r0  = float(np.linalg.norm(tip - goal_xyz_fixed))
    min_r = r0
    path_len = 0.0

    bids = subchain_ids(model, "B12", "B23")

    T_BACK_START  = T_APPROACH
    T_FRONT_START = T_BACK_START + BACKROLL_SEC
    T_DWELL_START = T_FRONT_START + FRONTROLL_SEC
    T_GRASP_START = T_DWELL_START + ROLL_DWELL_SEC

    grasp_step_idx = 0
    grasp_started = False

    for k in range(seq_u.shape[0]):
        u_k = seq_u[k].copy()
        for _ in range(steps_per):
            t = d.time
            if t < T_BACK_START:
                u_eff = u_k
            elif t < T_FRONT_START:
                prog = (t - T_BACK_START) / max(BACKROLL_SEC, 1e-9)
                u_eff = fast_roll_profile(prog, ROLLER_IDX_BACK, keep_other_low=True)
            elif t < T_DWELL_START:
                prog = (t - T_FRONT_START) / max(FRONTROLL_SEC, 1e-9)
                u_eff = fast_roll_profile(prog, ROLLER_IDX_FRONT, keep_other_low=False)
            elif t < T_GRASP_START:
                u_eff = fast_roll_profile(1.0, ROLLER_IDX_FRONT, keep_other_low=False)
            else:
                if not grasp_started:
                    grasp_started = True
                    grasp_step_idx = 0
                if grasp_step_idx < GRASP_RAMP_STEPS:
                    u_eff = perfect_grasp_control(ROLLER_IDX_FRONT, grasp_step_idx)
                    grasp_step_idx += 1
                else:
                    u_eff = perfect_grasp_control(ROLLER_IDX_FRONT, GRASP_RAMP_STEPS - 1)

            p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)

            last = tip.copy()
            mujoco.mj_step(model, d)
            tip = d.site_xpos[ee_sid].copy()
            r = float(np.linalg.norm(tip - goal_xyz_fixed))
            min_r = min(min_r, r)
            path_len += float(np.linalg.norm(tip - last))

    cov = coverage_score(d, bids, goal_xyz_fixed)
    area, fill = area_fill_score(d, bids, goal_xyz_fixed, rays=24)

    final_r_now = float(np.linalg.norm(d.site_xpos[ee_sid] - goal_xyz_fixed))
    miss = max(final_r_now - HOLD_TARGET_R, 0.0)
    near_term = max(min_r - HOLD_TARGET_R, 0.0)
    stall_pen = max(0.05 - path_len, 0.0)

    cost = (2.0*miss + 1.0*near_term
            - 1.6*cov - 0.55*area - 0.35*fill
            + 0.8*stall_pen)
    return float(cost)

def optimize_once(model, ee_sid, goal_xyz_fixed, iters=15, sigma=0.35, pop=None):
    dim = (1+2*N_HARM) + (1+2*N_HARM) + 5 + 5
    x0 = default_coeffs()
    if pop is None:
        pop = max(60, 3*dim)
    es = CMAEvolutionStrategy(x0, sigma, {'popsize': pop})

    best_cost = 1e9
    best_x = x0.copy()
    for it in range(iters):
        X = es.ask()
        F = []
        for x in X:
            c = cost_right_only(model, ee_sid, goal_xyz_fixed, x)
            F.append(c)
            if c < best_cost:
                best_cost = c
                best_x = x.copy()
        es.tell(X, F)
        if (it % 2) == 0:
            print(f"[CMA] it {it+1}/{iters} best_cost={best_cost:.6f}")
    return best_x, float(best_cost)


# ========================= Episode rollout (multi) =========================
def rollout_episode(model, ee_sid, goal_xyz_fixed, coeffs, plan,
                    obj_geom_ids, obj_friction_backup,
                    show=True, ep_idx=1):

    d = mujoco.MjData(model)

    # Deterministic reset
    reset_to_key(model, d, KEY_RIGHT_NAME)
    set_object_pose_freejoint(model, d, goal_xyz_fixed, cylinder_face_on_quat_wxyz())
    restore_geom_friction(model, obj_friction_backup)

    # Solver robustness
    try:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance  = min(model.opt.tolerance, 1e-9)

    # Episode knobs
    mode = str(plan.get("mode", "throw")).lower().strip()
    if mode not in ("hold", "throw"):
        mode = "throw"

    third_dir = str(plan.get("third_dir", "back")).lower().strip()
    if third_dir not in ("back", "front"):
        third_dir = "back"
    third_roller_idx = ROLLER_IDX_BACK if third_dir == "back" else ROLLER_IDX_FRONT

    thirdroll_sec = float(plan.get("thirdroll_sec", BASE_THIRDROLL_SEC))
    thirdroll_sec = float(np.clip(thirdroll_sec, 0.90, 3.50))  # velocity knob

    third_keep_other_low = bool(plan.get("third_keep_other_low", False))

    # Release scheduling (only used in throw mode)
    extra_hold_tail = float(plan.get("extra_hold_tail", 0.0))
    release_after   = float(plan.get("release_after", 0.0))
    release_ramp_s  = float(plan.get("release_ramp", 0.20))
    release_ramp_s  = float(np.clip(release_ramp_s, 0.05, 0.60))

    release_fric_scale = float(plan.get("release_friction_scale", 0.05))
    release_fric_scale = float(np.clip(release_fric_scale, 0.001, 1.0))

    unwind_sec = float(plan.get("unwind_sec", 0.55))
    unwind_sec = float(np.clip(unwind_sec, 0.0, 1.50))

    unwrap_dir = str(plan.get("unwrap_dir", "front")).lower().strip()
    if unwrap_dir not in ("front", "back"):
        unwrap_dir = "front"
    unwrap_strength = float(plan.get("unwrap_strength", 0.20))

    # Phase times
    T_BACK_START  = T_APPROACH
    T_FRONT_START = T_BACK_START  + BACKROLL_SEC
    T_DWELL_START = T_FRONT_START + FRONTROLL_SEC
    T_GRASP_START = T_DWELL_START + ROLL_DWELL_SEC
    T_THIRD_START = T_GRASP_START + GRASP_RAMP_SEC
    T_THIRD_END   = T_THIRD_START + thirdroll_sec

    # Release window (throw mode)
    T_POST_START    = T_THIRD_END
    T_REL_START     = T_POST_START + extra_hold_tail + release_after
    T_REL_END       = T_REL_START + release_ramp_s

    # Explicit unwind to peel off after release
    T_UNWIND_START  = T_REL_END
    T_UNWIND_END    = T_UNWIND_START + unwind_sec

    # Shake window
    T_SHAKE_START   = T_UNWIND_END
    T_SHAKE_END     = T_SHAKE_START + UNWRAP_SEC

    # End time
    if mode == "hold":
        T_END = T_THIRD_END + 0.80
    else:
        T_END = T_SHAKE_END + POST_RELEASE_TAIL_S

    # Build CMA control only up to max(T_END)
    seq_u = control_trajectory_from_coeffs(T_END, CTRL_HZ, coeffs, dir_sign=+1.0)

    # Simulation steps
    steps_total = int(math.ceil(T_END / DT_SIM))

    nu = model.nu
    p = np.full(nu, INIT_PRESSURE, float)
    p_smooth = p.copy()

    viewer = SafeViewer(model, d, show=show)

    # Logs
    u_eff_log = []
    ctrl_log  = []
    r_log = []
    th_log = []

    bids = subchain_ids(model, "B12", "B23")

    mujoco.mj_forward(model, d)
    tip = d.site_xpos[ee_sid].copy()
    min_r = float(np.linalg.norm(tip - goal_xyz_fixed))
    final_r = min_r
    path_len = 0.0

    # Grasp ramp internal
    grasp_started = False
    grasp_step_idx = 0

    # Release state
    friction_lowered = False
    friction_restored = False
    release_sampled = False
    release_velocity_xz = np.zeros(2, float)
    release_pos_xz = np.zeros(2, float)

    # Unwind direction: opposite of roll3 roller for peel
    unwind_dir = "front" if third_dir == "back" else "back"
    unwind_roller_idx = ROLLER_IDX_FRONT if unwind_dir == "front" else ROLLER_IDX_BACK

    try:
        for step in range(steps_total):
            t = d.time

            # CMA index
            k = min(int(t * CTRL_HZ), seq_u.shape[0] - 1)
            u_k = seq_u[k].copy()

            # --- Phase 0: CMA approach ---
            if t < T_BACK_START:
                u_eff = u_k

            # --- Phase 1: Roll1 BACK (pure roll) ---
            elif t < T_FRONT_START:
                prog = (t - T_BACK_START) / max(BACKROLL_SEC, 1e-9)
                u_eff = fast_roll_profile(prog, ROLLER_IDX_BACK, keep_other_low=True)

            # --- Phase 2: Roll2 FRONT (catch/hold bias) ---
            elif t < T_DWELL_START:
                prog = (t - T_FRONT_START) / max(FRONTROLL_SEC, 1e-9)
                u_eff = fast_roll_profile(prog, ROLLER_IDX_FRONT, keep_other_low=False)

            # --- Phase 3: Dwell ---
            elif t < T_GRASP_START:
                u_eff = fast_roll_profile(1.0, ROLLER_IDX_FRONT, keep_other_low=False)

            # --- Phase 4+: Grasp ramp then roll3 then post ---
            else:
                if not grasp_started:
                    grasp_started = True
                    grasp_step_idx = 0

                # 4a) Grasp ramp
                if grasp_step_idx < GRASP_RAMP_STEPS:
                    u_eff = perfect_grasp_control(ROLLER_IDX_FRONT, grasp_step_idx)
                    grasp_step_idx += 1

                else:
                    u_hold = perfect_grasp_control(ROLLER_IDX_FRONT, GRASP_RAMP_STEPS - 1)
                    u_release = np.array([0.0, 0.0], float)

                    # 5) Roll3 (episode-dependent)
                    if t < T_THIRD_END:
                        prog3 = (t - T_THIRD_START) / max(thirdroll_sec, 1e-9)
                        u_eff = fast_roll_profile(prog3, third_roller_idx, keep_other_low=third_keep_other_low)

                    else:
                        # HOLD mode: never release
                        if mode == "hold":
                            u_eff = u_hold

                        # THROW mode: hold -> release -> unwind -> shake
                        else:
                            if t < T_REL_START:
                                # holding window
                                u_eff = u_hold if (not third_keep_other_low) else np.array([0.15, 0.15], float)

                            elif t < T_REL_END:
                                # Drop friction + ramp down to zero
                                if not friction_lowered:
                                    set_geom_friction_scale_from_backup(
                                        model, obj_geom_ids, obj_friction_backup, release_fric_scale
                                    )
                                    friction_lowered = True

                                frac = (t - T_REL_START) / max((T_REL_END - T_REL_START), 1e-9)
                                u_eff = lerp_smooth(u_hold, u_release, frac)

                            elif t < T_UNWIND_END:
                                # UNWIND peel-off (pure roll, other low)
                                frac = (t - T_UNWIND_START) / max((T_UNWIND_END - T_UNWIND_START), 1e-9)
                                u_eff = fast_roll_profile(frac, unwind_roller_idx, keep_other_low=True)

                            elif t < T_SHAKE_END:
                                # Shake assist, still low-ish
                                u_eff = unwrap_command(t, T_SHAKE_START, T_SHAKE_END, unwrap_dir, unwrap_strength)

                            else:
                                # Restore friction after we’re done
                                if friction_lowered and (not friction_restored):
                                    restore_geom_friction(model, obj_friction_backup)
                                    friction_restored = True
                                u_eff = u_release

            # Apply pressure dynamics
            p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)

            # Log
            u_eff_log.append(u_eff.copy())
            ctrl_log.append(d.ctrl.copy())

            # Step
            last_tip = tip.copy()
            mujoco.mj_step(model, d)
            tip = d.site_xpos[ee_sid].copy()

            if (mode == "throw") and (not release_sampled) and (d.time >= T_REL_END):
                release_velocity_xz = object_freejoint_qvel_xz(model, d)
                oc = get_object_center_from_state(model, d)
                release_pos_xz = np.array([oc[0], oc[2]], float)
                release_sampled = True

            r, th = polar_of(tip, goal_xyz_fixed)
            r_log.append(r); th_log.append(th)
            path_len += float(np.linalg.norm(tip - last_tip))
            min_r = min(min_r, r)
            final_r = r

            viewer.draw()
            time.sleep(DT_SIM * max(PLAYBACK_SLOWDOWN, 0.0))

    finally:
        time.sleep(max(POST_VIEW_SEC, 0.0))
        viewer.close()
        # Safety restore (in case we exit early)
        restore_geom_friction(model, obj_friction_backup)

    # Metrics snapshot
    cov = coverage_score(d, bids, goal_xyz_fixed)
    area, fill = area_fill_score(d, bids, goal_xyz_fixed, rays=24)

    r_arr = np.asarray(r_log)
    th_arr = np.asarray(th_log)
    near_mask = (r_arr <= NEAR_R + 1e-9)
    if np.any(near_mask):
        th_unwrap = unwrap_angle_seq(th_arr[near_mask])
        w = np.diff(th_unwrap, prepend=th_unwrap[:1]) / DT_SIM
        omega_mean = float(np.mean(np.abs(w)))
        omega_peak = float(np.max(np.abs(w)))
    else:
        omega_mean = 0.0
        omega_peak = 0.0

    diag = dict(
        ep=int(ep_idx),
        plan_name=str(plan.get("name", f"ep{ep_idx}")),
        mode=mode,
        third_dir=third_dir,
        thirdroll_sec=float(thirdroll_sec),
        third_keep_other_low=bool(third_keep_other_low),
        final_r=float(final_r),
        min_r=float(min_r),
        cov_score=float(cov),
        area_m2=float(area),
        fill01=float(fill),
        omega_near_mean=float(omega_mean),
        omega_near_peak=float(omega_peak),
        path_len=float(path_len),
        T_END=float(T_END),
        release_friction_scale=(None if mode == "hold" else float(release_fric_scale)),
        unwind_sec=(None if mode == "hold" else float(unwind_sec)),
        release_velocity_xz=release_velocity_xz.tolist(),
        release_speed=float(np.linalg.norm(release_velocity_xz)),
        release_pos_xz=release_pos_xz.tolist(),
    )

    if "desired_dir_xz" in plan:
        ddes = normalize_xz(plan["desired_dir_xz"])
        v = release_velocity_xz
        speed = float(np.linalg.norm(v))
        diag["desired_dir_xz"] = ddes.tolist()
        diag["desired_speed"] = float(plan.get("desired_speed", 0.0))
        diag["dir_cos"] = float(np.dot(v, ddes) / max(speed, 1e-9))
        diag["speed_error"] = float(abs(speed - float(plan.get("desired_speed", 0.0))))


    return diag, np.asarray(u_eff_log, float), np.asarray(ctrl_log, float)


# ========================= MAIN =========================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = DT_SIM

    ee_sid = site_id(model, EE_SITE_NAME, default_last=True)

    # friction backup for the object
    obj_geom_ids = geom_ids_for_body(model, OBJ_BODY_NAME)
    obj_friction_backup = snapshot_geom_friction(model, obj_geom_ids)
    print(f"[info] O1 geom ids = {obj_geom_ids}")

    # Fixed object coordinate from keyframe reset state
    d0 = mujoco.MjData(model)
    reset_to_key(model, d0, KEY_RIGHT_NAME)
    goal_xyz_fixed = get_object_center_from_state(model, d0)

    print(f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep:.4f}")
    print(f"EE site: {EE_SITE_NAME}")
    print(f"Fixed object xyz: {goal_xyz_fixed}")
    print(f"Roll2 FRONT duration: {FRONTROLL_SEC:.2f}s (complete)")
    print(f"Roll3 base duration : {BASE_THIRDROLL_SEC:.2f}s (episodes override for velocity/direction)")
    print(f"Episodes: {len(EPISODE_PLANS)}")

    # Optimize once for approach+wrap+grasp
    dim = (1+2*N_HARM) + (1+2*N_HARM) + 5 + 5
    pop = max(60, 3*dim)

    coeffs_best, best_cost = optimize_once(
        model, ee_sid, goal_xyz_fixed,
        iters=15, sigma=0.35, pop=pop
    )
    np.save(os.path.join(SAVE_DIR, "best_coeffs.npy"), coeffs_best)
    print(f"[CMA] best_cost = {best_cost:.6f}")

    summary = {
        "XML_PATH": XML_PATH,
        "DT_SIM": DT_SIM,
        "CTRL_HZ": CTRL_HZ,
        "goal_xyz_fixed": goal_xyz_fixed.tolist(),
        "timing": dict(
            T_APPROACH=T_APPROACH,
            BACKROLL_SEC=BACKROLL_SEC,
            FRONTROLL_SEC=FRONTROLL_SEC,
            ROLL_DWELL_SEC=ROLL_DWELL_SEC,
            GRASP_RAMP_SEC=GRASP_RAMP_SEC,
            BASE_THIRDROLL_SEC=BASE_THIRDROLL_SEC,
            UNWRAP_SEC=UNWRAP_SEC,
        ),
        "best_cost": float(best_cost),
        "episodes": []
    }

    for i, plan in enumerate(EPISODE_PLANS, start=1):
        print(f"\n=== EP{i}: {plan.get('name','(no name)')} | mode={plan.get('mode')} ===")

        # show all episodes (change to show=(i==1) if you want only first displayed)
        diag, u_eff_log, ctrl_applied = rollout_episode(
            model, ee_sid, goal_xyz_fixed, coeffs_best, plan,
            obj_geom_ids=obj_geom_ids,
            obj_friction_backup=obj_friction_backup,
            show=True,
            ep_idx=i
        )

        ctrl_file = f"episode_ep{i}_RIGHT_ctrl.npy"
        u_file    = f"episode_ep{i}_RIGHT_u_eff.npy"
        np.save(os.path.join(SAVE_DIR, ctrl_file), ctrl_applied)
        np.save(os.path.join(SAVE_DIR, u_file), u_eff_log)

        print(f"[EP{i}] final_r={diag['final_r']:.4f} min_r={diag['min_r']:.4f} "
              f"cov={diag['cov_score']:.3f} fill={diag['fill01']:.3f} "
              f"ω_mean={diag['omega_near_mean']:.3f} ω_peak={diag['omega_near_peak']:.3f} "
              f"path={diag['path_len']:.3f} T_END={diag['T_END']:.2f}")

        summary["episodes"].append({
            "ep": i,
            "plan": plan,
            "diag": diag,
            "ctrl_file": ctrl_file,
            "u_eff_file": u_file,
            "n_steps": int(ctrl_applied.shape[0]),
        })

    with open(os.path.join(SAVE_DIR, "multi_episode_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {SAVE_DIR}/")
    print("Key files: best_coeffs.npy, multi_episode_summary.json, episode_ep*_RIGHT_ctrl.npy")



# ========================= CMA throw-plan optimization =========================
def sigmoid01(x):
    return 1.0 / (1.0 + np.exp(-float(x)))

def lerp_bound(name, z):
    lo, hi = THROW_PARAM_BOUNDS[name]
    return float(lo + (hi - lo) * sigmoid01(z))

def decode_throw_params(x, target):
    """Map unconstrained CMA vector to a rollout plan."""
    x = np.asarray(x, float).reshape(-1)
    # x[0] is a continuous side selector; CMA can choose the roller direction.
    third_dir = "front" if x[0] >= 0.0 else "back"
    unwrap_dir = "back" if third_dir == "front" else "front"
    return dict(
        name=str(target["name"]),
        mode="throw",
        desired_dir_xz=tuple(target["desired_dir_xz"]),
        desired_speed=float(target["desired_speed"]),
        third_dir=third_dir,
        thirdroll_sec=lerp_bound("thirdroll_sec", x[1]),
        third_keep_other_low=False,
        extra_hold_tail=lerp_bound("extra_hold_tail", x[2]),
        release_after=lerp_bound("release_after", x[3]),
        release_ramp=lerp_bound("release_ramp", x[4]),
        release_friction_scale=lerp_bound("release_friction_scale", x[5]),
        unwind_sec=lerp_bound("unwind_sec", x[6]),
        unwrap_dir=unwrap_dir,
        unwrap_strength=lerp_bound("unwrap_strength", x[7]),
    )

def throw_cost_from_diag(diag, target):
    v = np.asarray(diag.get("release_velocity_xz", [0.0, 0.0]), float)
    speed = float(np.linalg.norm(v))
    ddes = normalize_xz(target["desired_dir_xz"])
    dir_cos = float(np.dot(v, ddes) / max(speed, 1e-9))
    # CMA-ES minimizes. 0 is perfect direction; speed error is normalized.
    dir_cost = 1.0 - dir_cos
    desired_speed = float(target["desired_speed"])
    speed_cost = ((speed - desired_speed) / max(desired_speed, 1e-6)) ** 2
    release_cost = 0.0 if speed > 1e-4 else 1.0
    # Keep the grasp/coverage term as a soft constraint; do not let it dominate the throw.
    grasp_cost = max(diag["final_r"] - HOLD_TARGET_R, 0.0) - 0.25 * diag["cov_score"] - 0.10 * diag["fill01"]
    time_cost = float(diag["T_END"])
    return float(W_DIR*dir_cost + W_SPEED*speed_cost + W_GRASP*grasp_cost + W_RELEASE*release_cost + W_TIME*time_cost)

def cost_throw_params(model, ee_sid, goal_xyz_fixed, coeffs, target, x, obj_geom_ids, obj_friction_backup):
    plan = decode_throw_params(x, target)
    diag, _, _ = rollout_episode(
        model, ee_sid, goal_xyz_fixed, coeffs, plan,
        obj_geom_ids=obj_geom_ids,
        obj_friction_backup=obj_friction_backup,
        show=False,
        ep_idx=0,
    )
    return throw_cost_from_diag(diag, target)

def optimize_throw_plan(model, ee_sid, goal_xyz_fixed, coeffs, target, obj_geom_ids, obj_friction_backup,
                        iters=18, sigma=0.85, pop=32):
    # [direction selector, thirdroll_sec, hold_tail, release_after, release_ramp,
    #  friction_scale, unwind_sec, unwrap_strength]
    x0 = np.zeros(8, float)
    es = CMAEvolutionStrategy(x0, sigma, {"popsize": int(pop)})
    best_x = x0.copy()
    best_cost = 1e18
    for it in range(int(iters)):
        X = es.ask()
        F = []
        for x in X:
            c = cost_throw_params(model, ee_sid, goal_xyz_fixed, coeffs, target, x, obj_geom_ids, obj_friction_backup)
            F.append(c)
            if c < best_cost:
                best_cost = float(c)
                best_x = np.asarray(x, float).copy()
        es.tell(X, F)
        if (it % 2) == 0:
            print(f"[CMA-THROW:{target['name']}] it {it+1}/{iters} best={best_cost:.5f}")
    return decode_throw_params(best_x, target), float(best_cost), best_x


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = DT_SIM
    ee_sid = site_id(model, EE_SITE_NAME, default_last=True)

    obj_geom_ids = geom_ids_for_body(model, OBJ_BODY_NAME)
    obj_friction_backup = snapshot_geom_friction(model, obj_geom_ids)
    print(f"[info] O1 geom ids = {obj_geom_ids}")

    d0 = mujoco.MjData(model)
    reset_to_key(model, d0, KEY_RIGHT_NAME)
    goal_xyz_fixed = get_object_center_from_state(model, d0)

    print(f"Loaded: nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={model.opt.timestep:.4f}")
    print(f"Fixed object xyz: {goal_xyz_fixed}")
    print(f"Episode targets: {len(EPISODE_TARGETS)} multidirection/multispeed throws")

    dim = (1+2*N_HARM) + (1+2*N_HARM) + 5 + 5
    coeffs_best, best_cost = optimize_once(
        model, ee_sid, goal_xyz_fixed,
        iters=15, sigma=0.35, pop=max(60, 3*dim)
    )
    np.save(os.path.join(SAVE_DIR, "best_grasp_coeffs.npy"), coeffs_best)
    print(f"[CMA-GRASP] best_cost = {best_cost:.6f}")

    summary = {
        "XML_PATH": XML_PATH,
        "SAVE_DIR": SAVE_DIR,
        "DT_SIM": DT_SIM,
        "CTRL_HZ": CTRL_HZ,
        "goal_xyz_fixed": goal_xyz_fixed.tolist(),
        "best_grasp_cost": float(best_cost),
        "throw_cost_weights": dict(W_DIR=W_DIR, W_SPEED=W_SPEED, W_GRASP=W_GRASP, W_RELEASE=W_RELEASE, W_TIME=W_TIME),
        "episodes": [],
    }

    for i, target in enumerate(EPISODE_TARGETS, start=1):
        print(f"\n=== Target EP{i}: {target['name']} dir={target['desired_dir_xz']} speed={target['desired_speed']:.3f} ===")
        plan, plan_cost, raw_x = optimize_throw_plan(
            model, ee_sid, goal_xyz_fixed, coeffs_best, target,
            obj_geom_ids, obj_friction_backup,
            iters=18, sigma=0.85, pop=32,
        )
        np.save(os.path.join(SAVE_DIR, f"episode_ep{i}_throw_cma_x.npy"), raw_x)

        diag, u_eff_log, ctrl_applied = rollout_episode(
            model, ee_sid, goal_xyz_fixed, coeffs_best, plan,
            obj_geom_ids=obj_geom_ids,
            obj_friction_backup=obj_friction_backup,
            show=True,
            ep_idx=i,
        )

        ctrl_file = f"episode_ep{i}_RIGHT_ctrl.npy"
        u_file = f"episode_ep{i}_RIGHT_u_eff.npy"
        np.save(os.path.join(SAVE_DIR, ctrl_file), ctrl_applied)
        np.save(os.path.join(SAVE_DIR, u_file), u_eff_log)

        print(f"[EP{i}] target_speed={target['desired_speed']:.3f} release_speed={diag['release_speed']:.3f} "
              f"dir_cos={diag.get('dir_cos', 0.0):.3f} speed_err={diag.get('speed_error', 0.0):.3f} "
              f"third_dir={plan['third_dir']} thirdroll={plan['thirdroll_sec']:.2f}s cost={plan_cost:.4f}")

        summary["episodes"].append({
            "ep": i,
            "target": target,
            "optimized_plan": plan,
            "throw_cma_cost": float(plan_cost),
            "diag": diag,
            "ctrl_file": ctrl_file,
            "u_eff_file": u_file,
            "n_steps": int(ctrl_applied.shape[0]),
        })

    with open(os.path.join(SAVE_DIR, "multi_direction_multi_speed_cma_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved outputs to: {SAVE_DIR}/")
    print("Key files: best_grasp_coeffs.npy, multi_direction_multi_speed_cma_summary.json, episode_ep*_RIGHT_ctrl.npy")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
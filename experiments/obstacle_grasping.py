#!/usr/bin/env python3
# test_obstacles.py
#
# RIGHT: CMA-optimized control with roll + glide + grasp shaping, SHOW + RECORD.
# LEFT:  strict mirrored replay with channel swap, SHOW + RECORD.
#
# Behavior (same structure as smooth non-obstacle version), plus:
#   - Cylinder O1 positioned via free joint `O1_freejoint`.
#   - Obstacle cylinder O2 is a mocap body, placed beside O1 via a deterministic rule.
#
# Outputs per episode:
#   - ab_pair_ep{N}.npy
#   - episode_ep{N}_RIGHT_ctrl.npy
#   - episode_ep{N}_LEFT_ctrl.npy
#   - full_mirror_AB_summary.json

import os, time, math, json, random
import numpy as np
import multiprocessing as mp

# Thread caps (determinism-ish)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")  # ensure a GL backend

import mujoco
from cma import CMAEvolutionStrategy

try:
    import mujoco.viewer as mjv
    HAVE_MJV = True
except Exception:
    HAVE_MJV = False

# -------- Paths & IDs --------
XML_PATH           = str(Path(__file__).resolve().parents[1] / "SpiRob_obstacle.xml")
EE_SITE_NAME       = "s23_3"
CYL_FREEJOINT_NAME = "O1_freejoint"
OBS_BODY_NAME      = "O2"            # mocap obstacle body

SAVE_DIR  = str(Path(__file__).resolve().parents[1] / "outputs" / "obstacles"); os.makedirs(SAVE_DIR, exist_ok=True)
SEED      = 123
np.random.seed(SEED); random.seed(SEED)

# -------- Goals (unchanged) --------
GOAL_XZ = [
    (0.1931,0.6790),(-0.0746,0.7827),(0.2229,0.5106),(0.1963,0.6327),
    (0.0858,0.7581),(-0.1978,0.6585),(-0.2150,0.5573),(-0.1964,0.6079),
    (-0.0821,0.6885),(-0.1767,0.7049),(-0.0675,0.7182),(0.2055,0.7085),
    (0.1574,0.6982),(0.1945,0.5649),(-0.1408,0.6714),(0.1708,0.6500),
    (0.0812,0.7177),(-0.0949,0.7324),(0.1082,0.7060),(0.1719,0.6099),
    (0.0645,0.6948),(0.2235,0.6804),(-0.1637,0.7413),(-0.1308,0.7385),
    (-0.2097,0.7142),(0.1110,0.7756),(-0.1649,0.6549),(0.2206,0.5779),
    (-0.2270,0.6230),(0.1546,0.7459),(0.1337,0.7242),(0.2255,0.6399),
    (-0.2084,0.5150),(-0.1224,0.7653),(0.2271,0.6085),(0.1418,0.6426),
    (-0.1123,0.6749),(-0.2275,0.6847),(0.0712,0.7902),(0.1993,0.6012),
    (-0.1106,0.7044),(0.0941,0.6776),
]

# -------- Geometry / targets (CYL radius = 0.015) --------
CYL_R            = 0.015
HOLD_TARGET_R    = CYL_R + 0.014
NEAR_R           = CYL_R + 0.026       # where contact wrapping is “active”

# Earlier blending start for smoother override fade-in
ROLL_NEAR_R      = NEAR_R + 0.016      # smoother than +0.010

SUBCHAIN_LEN_M   = 0.1071
CIRCUMFERENCE_M  = 2*math.pi*CYL_R
WRAPS_FEASIBLE   = SUBCHAIN_LEN_M / max(CIRCUMFERENCE_M, 1e-9)
ANG_TARGET_RAD   = 2*math.pi * min(1.0, WRAPS_FEASIBLE)

# Coverage band
COVER_BAND_IN    = CYL_R - 0.002
COVER_BAND_OUT   = CYL_R + 0.009
COVER_MIN_COUNT  = 9
COVER_MIN_SPAN   = math.radians(300.0)

# -------- Sim / actuation --------
DT_SIM            = 0.002
HORIZON_T         = 3.50
CTRL_HZ           = 220.0
PLAYBACK_SLOWDOWN = 2.0
POST_VIEW_SEC     = 1.0

INIT_PRESSURE     = 0.10
PRESSURE_TAU      = 0.10

# Much smoother than earlier versions
PRESSURE_SLEW     = 1.10    # lower = slower pressure changes
SMOOTH_ALPHA      = 0.08    # lower = heavier low-pass filtering

# Near-contact shaping (kept but slightly softened)
NEAR_GAIN_CENTER  = 1.25
NEAR_GAIN_DIFF    = 1.65

# Pre-contact approach shaping (Phase 3) — softened
APPROACH_GAIN_CENTER = 1.60
APPROACH_GAIN_DIFF   = 0.08
APPROACH_SLEW_MULT   = 1.7

# Roll unlock gate
ROLL_UNLOCK_MARGIN = 0.005
ROLL_UNLOCK_WIDTH  = 0.006
def smooth_gate(x, w): return 0.5*(1.0 + math.tanh(x / max(w, 1e-6)))

# Phase-1 stretch
PHASE1_EXTRA_FRAC = 0.20

# Backward + forward roll parameters
ROLL_TARGET_FRAC      = 1.10
BACKROLL_TARGET_FRAC  = ROLL_TARGET_FRAC
FRONTROLL_TARGET_FRAC = ROLL_TARGET_FRAC
BACKROLL_TARGET_RAD   = 2*math.pi*BACKROLL_TARGET_FRAC
FRONTROLL_TARGET_RAD  = 2*math.pi*FRONTROLL_TARGET_FRAC

PHASE_MIN_WINDOW      = 0.04
ROLL_DWELL_SEC        = 0.12
ROLL_MIN_DIFF         = 0.01
ROLL_SIGN_BY_DIR      = True   # direction-aware integration

# Fourier base + forced 5-phase
N_HARM = 2

# -------- Helper: blend between CMA cmd & override --------
def blend_u(u_base, u_override, w):
    w = float(np.clip(w, 0.0, 1.0))
    return (1.0 - w) * u_base + w * u_override

# -------- Param packing / control trajectory --------
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

    kappa   = 0.55 + 0.15*np.tanh(x[idx]); idx += 1
    epsilon = 0.04 * np.tanh(x[idx]);      idx += 1

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

    # enforce symmetry
    kappa   = 0.50
    epsilon = 0.0

    return (b0, bc, bs, s0, sc, ss, phase, kappa, epsilon, warp_k, warp_c,
            s1, s2, s3, s4, w, aF1, aB1, aF2, aB2, aF3)

def build_time_warp(N, warp_k, warp_c):
    τ_raw = np.linspace(0.0, 1.0, N, endpoint=True)
    β = 6.0
    speed = np.exp(warp_k * np.tanh(β*(τ_raw - warp_c)))
    speed = np.clip(speed, 1e-3, None)
    s = np.cumsum(speed)
    return (s - s[0]) / (s[-1] - s[0] + 1e-12)

def _stretch_phase1(s1, s2, s3, s4, extra_frac, min_win):
    l1 = max(s1 - 0.0, 1e-6)
    l2 = max(s2 - s1, 1e-6)
    l3 = max(s3 - s2, 1e-6)
    l4 = max(s4 - s3, 1e-6)
    l5 = max(1.0 - s4, 1e-6)
    total = l1 + l2 + l3 + l4 + l5
    borrowable = max((l2 - min_win), 0.0) + max((l3 - min_win), 0.0) + \
                 max((l4 - min_win), 0.0) + max((l5 - min_win), 0.0)
    extra = min(extra_frac * total, borrowable)
    if extra <= 1e-9:
        return s1, s2, s3, s4
    tail = l2 + l3 + l4 + l5
    def red(li): return extra * (li / max(tail, 1e-12))
    l1n = l1 + extra
    l2n = max(l2 - red(l2), min_win)
    l3n = max(l3 - red(l3), min_win)
    l4n = max(l4 - red(l4), min_win)
    l5n = max(l5 - red(l5), min_win)
    scale = total / (l1n + l2n + l3n + l4n + l5n)
    l1n *= scale; l2n *= scale; l3n *= scale; l4n *= scale; l5n *= scale
    s1n = l1n
    s2n = s1n + l2n
    s3n = s2n + l3n
    s4n = s3n + l4n
    return float(s1n), float(s2n), float(s3n), float(s4n)

def smooth_steps(τ, s1, s2, s3, s4, w):
    def S(x): return 0.5*(1+np.tanh(x/(w+1e-6)))
    σ1 = 1 - S(τ - s1)
    σ2 = S(τ - s1) * (1 - S(τ - s2))
    σ3 = S(τ - s2) * (1 - S(τ - s3))
    σ4 = S(τ - s3) * (1 - S(τ - s4))
    σ5 = S(τ - s4)
    Z  = np.clip(σ1 + σ2 + σ3 + σ4 + σ5, 1e-6, None)
    return σ1/Z, σ2/Z, σ3/Z, σ4/Z, σ5/Z

def control_trajectory_from_coeffs(T, ctrl_hz, coeffs, dir_sign=+1.0):
    (b0, bc, bs, s0, sc, ss, phase, kappa, epsilon, warp_k, warp_c,
     s1, s2, s3, s4, w, aF1, aB1, aF2, aB2, aF3) = squash_params(coeffs)
    s1, s2, s3, s4 = _stretch_phase1(
        s1, s2, s3, s4, PHASE1_EXTRA_FRAC, PHASE_MIN_WINDOW
    )
    N = int(T * ctrl_hz)
    τ = build_time_warp(N, warp_k, warp_c)

    # base waveform
    b = b0 * np.ones_like(τ)
    for i in range(1, N_HARM+1):
        b += bc[i-1]*np.cos(2*math.pi*i*τ) + bs[i-1]*np.sin(2*math.pi*i*τ)
    base = 1.0/(1.0 + np.exp(-b))

    s = s0 * np.ones_like(τ)
    for i in range(1, N_HARM+1):
        ang = 2*math.pi*i*τ + phase
        s += sc[i-1]*np.cos(ang) + ss[i-1]*np.sin(ang)

    σ1, σ2, σ3, σ4, σ5 = smooth_steps(τ, s1, s2, s3, s4, w)
    sign = +1.0 if dir_sign >= 0 else -1.0
    env = sign*(+aF1*σ1 - aB1*σ2 + aF2*σ3 - aB2*σ4 + aF3*σ5)

    uL = base + 0.50*env
    uR = base - 0.50*env
    u  = np.stack([uL, uR], axis=1)

    phase_stack = np.stack([σ1, σ2, σ3, σ4, σ5], axis=0)
    phase_id = 1 + np.argmax(phase_stack, axis=0)
    return np.clip(u, 0.0, 1.0), phase_id

# ====== Roll + Glide + Grasp policy ======

A_ROLLER_HOLD      = 0.98
A_SUPPORT_BASE     = 0.12
A_OTHER_START      = 0.50
A_OTHER_FINAL      = 0.92
GLIDE_NEAR_START   = NEAR_R + 0.010
GLIDE_NEAR_END     = HOLD_TARGET_R + 0.003
GRASP_EQUALIZE     = 0.80   # used only in fallback

# Perfect grasp ramp (smoother than original)
GRASP_RAMP_SEC      = 0.50
GRASP_SUPPORT_START = 0.45
GRASP_SUPPORT_FINAL = 1.00
GRASP_ROLLER_START  = 0.93
GRASP_ROLLER_FINAL  = 1.00
GRASP_RAMP_STEPS    = max(1, int(GRASP_RAMP_SEC / DT_SIM))

def first_roll_override(progress01, roller_idx):
    progress01 = float(np.clip(progress01, 0.0, 1.0))
    roller = A_ROLLER_HOLD + (1.0 - A_ROLLER_HOLD) * (1.0 - np.exp(-6.0*progress01))
    other  = A_SUPPORT_BASE
    if roller_idx == 1:
        u = np.array([other, roller], float)
    else:
        u = np.array([roller, other], float)
    return np.clip(u, 0.0, 1.0)

def glide_pull_profile(r_now):
    if r_now >= GLIDE_NEAR_START:
        s = 0.0
    elif r_now <= GLIDE_NEAR_END:
        s = 1.0
    else:
        s = (GLIDE_NEAR_START - r_now) / max(GLIDE_NEAR_START - GLIDE_NEAR_END, 1e-9)
    return float(np.clip(A_OTHER_START*(1.0 - s) + A_OTHER_FINAL*s, 0.0, 1.0))

def post_roll_glide_control(r_now, roller_idx):
    a_other = glide_pull_profile(r_now)
    if roller_idx == 0:
        u = np.array([A_ROLLER_HOLD, a_other], float)
    else:
        u = np.array([a_other, A_ROLLER_HOLD], float)
    return np.clip(u, 0.0, 1.0)

def perfect_grasp_control(roller_idx, step_idx):
    t = float(np.clip(step_idx / max(GRASP_RAMP_STEPS, 1), 0.0, 1.0))
    support_level = GRASP_SUPPORT_START + t * (GRASP_SUPPORT_FINAL - GRASP_SUPPORT_START)
    roller_level  = GRASP_ROLLER_START  + t * (GRASP_ROLLER_FINAL  - GRASP_ROLLER_START)
    if roller_idx == 0:
        u = np.array([roller_level, support_level], float)
    else:
        u = np.array([support_level, roller_level], float)
    return np.clip(u, 0.0, 1.0)

# -------- IDs / helpers --------
def site_id(model, name, default_last=False):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0 and default_last:
        return model.nsite - 1
    if sid < 0:
        raise RuntimeError(f"Site not found: {name}")
    return sid

def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Body not found: {name}")
    return bid

def polar_of(vec, center):
    dx, dy, dz = vec - center
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

def clamp_ang_step(dth, max_step=0.18):
    # Smaller per-step angle → smoother accumulated roll
    return float(np.clip(dth, -max_step, +max_step))

def coverage_score(data, body_ids, goal_xyz):
    gx, gz = goal_xyz[0], goal_xyz[2]
    th, rr = [], []
    for bid in body_ids:
        x, _, z = data.xipos[bid]
        dx, dz = x - gx, z - gz
        th.append(math.atan2(dz, dx))
        rr.append(math.hypot(dx, dz))
    thetas = np.asarray(th); r = np.asarray(rr)
    near = (r >= max(COVER_BAND_IN,1e-6)) & (r <= COVER_BAND_OUT)
    cnt = int(np.sum(near))
    if cnt > 1:
        ang = np.sort((thetas[near] + 2*math.pi) % (2*math.pi))
        diffs = np.diff(np.concatenate([ang, ang[:1] + 2*math.pi]))
        span = 2*math.pi - np.max(diffs)
    else:
        span = 0.0
    mean_r = float(np.mean(r[near])) if cnt>0 else 1e9
    span_deg = span * 180.0 / math.pi
    score = 0.0
    score += min(cnt/max(COVER_MIN_COUNT,1),1.5) * 0.5
    score += min(span_deg/360.0,1.5) * 0.4
    return float(score), (cnt>=COVER_MIN_COUNT and span>=math.radians(300.0) and mean_r<=COVER_BAND_OUT)

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
    return float(area), fill01

def distal_curvature(model, data):
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"B{i}") for i in [21,22,23]]
    pts = np.array([data.xipos[i] for i in ids], float)
    v1 = pts[1] - pts[0]; v2 = pts[2] - pts[1]
    v1 = v1[[0,2]]; v2 = v2[[0,2]]
    n1 = max(np.linalg.norm(v1), 1e-9)
    n2 = max(np.linalg.norm(v2), 1e-9)
    cosang = float(np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0))
    return 1.0 - cosang

# -------- Cylinder + obstacle placement --------
def init_cylinder_pose(model, data, goal_xyz):
    """
    Place cylinder body O1 at goal_xyz using its free joint O1_freejoint,
    keeping the default orientation from qpos0.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CYL_FREEJOINT_NAME)
    if jid < 0:
        raise RuntimeError(f"Freejoint '{CYL_FREEJOINT_NAME}' not found in model.")
    adr = model.jnt_qposadr[jid]

    # assuming data has been reset to qpos0 before calling this
    quat0 = data.qpos[adr+3:adr+7].copy()
    data.qpos[adr:adr+3] = np.asarray(goal_xyz, float)
    data.qpos[adr+3:adr+7] = quat0
    mujoco.mj_forward(model, data)

def set_obstacle_pose_mocap(model, data, xyz, quat):
    """
    Place obstacle body O2 (mocap) at given pose.
    """
    bid = body_id(model, OBS_BODY_NAME)
    mocap_id = model.body_mocapid[bid]
    if mocap_id < 0:
        raise RuntimeError(f"Body {OBS_BODY_NAME} is not mocap in XML")
    data.mocap_pos[mocap_id]  = np.asarray(xyz, float)
    data.mocap_quat[mocap_id] = np.asarray(quat, float)
    mujoco.mj_forward(model, data)

def cylinder_face_on_quat():
    # Cylinder axis along +y, face normal along ±x; keep original "face-on" quat.
    s = math.sin(math.pi/4); c = math.cos(math.pi/4)
    return (c, s, 0, 0)

def compute_obstacle_pose(goal_xyz):
    """
    Compute a pose for the obstacle (O2) that is diagonally beside the goal cylinder,
    not blocking the direct line from the robot base to the cylinder.

    Strategy:
      - base ~ robot base position
      - v = goal - base
      - take perpendicular in x-z plane, normalize
      - place obstacle at ~3 * CYL_R along that perpendicular,
        with sign chosen so z-offset is positive.
    """
    base = np.array([0.0, 0.0, 0.5], float)  # approximate B0 base
    v = np.asarray(goal_xyz, float) - base
    vxz = np.array([v[0], v[2]], float)
    n = np.linalg.norm(vxz)
    if n < 1e-6:
        vxz[:] = np.array([1.0, 0.0])
        n = 1.0
    vxz /= n
    perp = np.array([-vxz[1], vxz[0]], float)
    if perp[1] < 0:
        perp *= -1.0
    d = CYL_R * 3.0  # ~4.5 cm away from cylinder center
    offset = np.array([perp[0]*d, 0.0, perp[1]*d], float)
    return np.asarray(goal_xyz, float) + offset

# -------- Viewer wrapper --------
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
                self.v   = self.ctx.__enter__()
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
            if self.kind == "mjv":
                self.v.sync()
            else:
                self.v.render()
        except Exception:
            pass
    def close(self):
        if self.v is None:
            return
        try:
            if self.kind == "mjv":
                self.ctx.__exit__(None, None, None)
            else:
                self.v.close()
        except Exception:
            pass

# --- pressure dynamics helper (we log the actual commands used) ---
def apply_pressure_step(d, u01, p, p_smooth, dt, slew_override=None):
    alpha   = dt / max(PRESSURE_TAU, 1e-6)
    desired = p + alpha*(u01 - p)
    max_step = (PRESSURE_SLEW if slew_override is None else slew_override) * dt
    p = np.clip(p + np.clip(desired - p, -max_step, max_step), 0.0, 1.0)
    p_smooth = (1.0 - SMOOTH_ALPHA) * p_smooth + SMOOTH_ALPHA * p
    d.ctrl[:] = p_smooth
    return p, p_smooth

# -------- RIGHT rollout: SHOW + RECORD --------
def rollout_right_and_record(model, ee_sid, goal_xyz, coeffs, show=True):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)

    # place cylinder and obstacle
    init_cylinder_pose(model, d, goal_xyz)
    obs_xyz = compute_obstacle_pose(goal_xyz)
    set_obstacle_pose_mocap(model, d, obs_xyz, cylinder_face_on_quat())

    try:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance  = min(model.opt.tolerance, 1e-9)

    bids = subchain_ids(model, "B12", "B23")
    seq_u, phase_id = control_trajectory_from_coeffs(
        HORIZON_T, CTRL_HZ, coeffs, dir_sign=+1.0
    )
    ctrl_dt   = 1.0/CTRL_HZ
    steps_per = max(1, int(ctrl_dt/DT_SIM))
    sleep_dt  = DT_SIM * max(PLAYBACK_SLOWDOWN, 0.0)

    nu = model.nu
    p = np.full(nu, INIT_PRESSURE, float); p_smooth = p.copy()

    viewer = SafeViewer(model, d, show=show)

    mujoco.mj_forward(model, d)
    tip = d.site_xpos[ee_sid].copy()
    r0  = float(np.linalg.norm(tip - goal_xyz))
    min_r = r0; final_r = r0; path_len = 0.0

    # logs
    u_eff_log         = []
    ctrl_applied_log  = []
    r_log             = []
    th_log            = []

    # Roll / grasp state
    back_started   = False
    back_done      = False
    front_started  = False
    front_done     = False
    dwell_steps_left = 0
    roll_back_cum  = 0.0
    roll_front_cum = 0.0
    th_prev_unwrap = None
    grasp_started  = False
    grasp_step_idx = 0

    # which tendon is roller for backward vs forward
    roller_idx_back  = 0  # backward roll: first tendon
    roller_idx_front = 1  # forward roll: second tendon (toward cylinder)

    N_ctrl = seq_u.shape[0]

    try:
        for k in range(N_ctrl):
            u_k = seq_u[k].copy()
            for _ in range(steps_per):
                r_now = float(np.linalg.norm(tip - goal_xyz))

                # -------- Far region: CMA only + approach shaping --------
                if r_now > NEAR_R + 0.002 and phase_id[k] == 3:
                    uc = 0.5*(u_k[0] + u_k[1])
                    ud = 0.5*(u_k[0] - u_k[1])
                    gate = smooth_gate((NEAR_R + ROLL_UNLOCK_MARGIN) - r_now, ROLL_UNLOCK_WIDTH)
                    ud *= gate
                    if r_now > NEAR_R:
                        ud = 0.0
                    uc = np.clip(uc * APPROACH_GAIN_CENTER, 0.0, 1.0)
                    ud = np.clip(ud * APPROACH_GAIN_DIFF,  -0.5, 0.5)
                    u_eff = np.array([uc + ud, uc - ud], dtype=float)
                    slew  = PRESSURE_SLEW * APPROACH_SLEW_MULT
                    p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM,
                                                      slew_override=slew)

                # -------- Near region: blended roll / glide / grasp --------
                elif r_now <= NEAR_R + 0.004:
                    _, th_now = polar_of(tip, goal_xyz)

                    if th_prev_unwrap is None:
                        th_prev_unwrap = th_now

                    dth_raw = (th_now - th_prev_unwrap + math.pi) % (2*math.pi) - math.pi
                    dth     = clamp_ang_step(dth_raw)
                    th_prev_unwrap = th_prev_unwrap + dth

                    # initialize state machine on first entry
                    if not back_started and not front_started:
                        back_started = True
                        roll_back_cum  = 0.0
                        roll_front_cum = 0.0
                        front_done     = False
                        grasp_started  = False
                        grasp_step_idx = 0
                        dwell_steps_left = 0

                    if back_started and not back_done:
                        roll_back_cum += max(0.0, abs(dth))
                        progressB = float(np.clip(
                            roll_back_cum / max(BACKROLL_TARGET_RAD, 1e-6), 0.0, 1.0
                        ))
                        u_state = first_roll_override(progressB, roller_idx_back)
                        if roll_back_cum >= BACKROLL_TARGET_RAD:
                            back_done      = True
                            front_started  = True
                            roll_front_cum = 0.0

                    elif front_started and not front_done:
                        roll_front_cum += max(0.0, abs(dth))
                        progressF = float(np.clip(
                            roll_front_cum / max(FRONTROLL_TARGET_RAD, 1e-6), 0.0, 1.0
                        ))
                        u_state = first_roll_override(progressF, roller_idx_front)
                        if roll_front_cum >= FRONTROLL_TARGET_RAD:
                            front_done      = True
                            dwell_steps_left = max(1, int(ROLL_DWELL_SEC / DT_SIM))

                    elif front_done and dwell_steps_left > 0 and r_now > HOLD_TARGET_R:
                        dwell_steps_left -= 1
                        u_state = post_roll_glide_control(r_now, roller_idx_front)

                    else:
                        if r_now > HOLD_TARGET_R:
                            u_state = post_roll_glide_control(r_now, roller_idx_front)
                        else:
                            if not grasp_started:
                                grasp_started  = True
                                grasp_step_idx = 0
                            u_state = perfect_grasp_control(roller_idx_front, grasp_step_idx)
                            grasp_step_idx += 1

                    # Smooth radial blending between CMA and state machine (S-curve)
                    lin = np.clip(
                        (ROLL_NEAR_R - r_now) / max(ROLL_NEAR_R - NEAR_R, 1e-6),
                        0.0, 1.0
                    )
                    w_r = 0.5 * (1.0 + np.tanh(3.0 * (lin - 0.5)))
                    u_eff = blend_u(u_k, u_state, w_r)
                    p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)

                # -------- Mid region (between NEAR_R and ROLL_NEAR_R) --------
                else:
                    u_eff = u_k
                    p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)

                # Logging
                u_eff_log.append(u_eff.copy())
                ctrl_applied_log.append(d.ctrl.copy())

                last_tip = d.site_xpos[ee_sid].copy()
                mujoco.mj_step(model, d)
                tip = d.site_xpos[ee_sid].copy()
                r, th = polar_of(tip, goal_xyz)
                r_log.append(r); th_log.append(th)
                path_len += float(np.linalg.norm(tip - last_tip))
                min_r = min(min_r, r); final_r = r

                viewer.draw()
                time.sleep(sleep_dt)

    finally:
        time.sleep(max(POST_VIEW_SEC, 0.0))
        viewer.close()

    # metrics
    cov_score, _ = coverage_score(d, bids, goal_xyz)
    area_m2, fill01 = area_fill_score(d, bids, goal_xyz, rays=24)

    r_arr = np.asarray(r_log); th_arr = np.asarray(th_log)
    near_mask = (r_arr <= NEAR_R + 1e-9)
    if np.any(near_mask):
        th_unwrap = unwrap_angle_seq(th_arr[near_mask])
        ω = np.diff(th_unwrap, prepend=th_unwrap[:1]) / DT_SIM
        omega_near_mean = float(np.mean(np.abs(ω)))
        omega_near_peak = float(np.max(np.abs(ω)))
    else:
        omega_near_mean = 0.0; omega_near_peak = 0.0

    diag = dict(
        final_r=float(final_r), min_r=float(min_r),
        area_m2=float(area_m2), fill01=float(fill01),
        path_len=float(path_len), cov_score=float(cov_score),
        roll1_done=True,  # “roll sequence done”
        omega_near_mean=omega_near_mean, omega_near_peak=omega_near_peak
    )
    return diag, np.array(u_eff_log, float), np.array(ctrl_applied_log, float)

# -------- LEFT replay (SHOWS) with channel swap + RECORD applied --------
def replay_left_with_u_log(model, ee_sid, goal_xyz_left, u_log, show=True):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)

    # place cylinder and obstacle for LEFT goal
    init_cylinder_pose(model, d, goal_xyz_left)
    obs_xyz = compute_obstacle_pose(goal_xyz_left)
    set_obstacle_pose_mocap(model, d, obs_xyz, cylinder_face_on_quat())

    try:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance  = min(model.opt.tolerance, 1e-9)

    viewer = SafeViewer(model, d, show=show)
    sleep_dt  = DT_SIM * max(PLAYBACK_SLOWDOWN, 0.0)

    nu = model.nu
    p = np.full(nu, INIT_PRESSURE, float); p_smooth = p.copy()

    mujoco.mj_forward(model, d)
    tip = d.site_xpos[ee_sid].copy()
    min_r = float(np.linalg.norm(tip - goal_xyz_left)); final_r = min_r; path_len = 0.0

    r_log = []; th_log = []
    ctrl_applied_log = []

    try:
        for u_step in u_log:
            # swap channels for perfect mirror (R↔L)
            u_eff = np.array([u_step[1], u_step[0]], float)
            p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)
            ctrl_applied_log.append(d.ctrl.copy())

            last_tip = d.site_xpos[ee_sid].copy()
            mujoco.mj_step(model, d)
            tip = d.site_xpos[ee_sid].copy()
            r, th = polar_of(tip, goal_xyz_left)
            r_log.append(r); th_log.append(th)
            min_r = min(min_r, r); final_r = r
            path_len += float(np.linalg.norm(tip - last_tip))

            viewer.draw()
            time.sleep(sleep_dt)
    finally:
        time.sleep(max(POST_VIEW_SEC, 0.0))
        viewer.close()

    # metrics
    bids = subchain_ids(model, "B12", "B23")
    cov_score, _ = coverage_score(d, bids, goal_xyz_left)
    area_m2, fill01 = area_fill_score(d, bids, goal_xyz_left, rays=24)

    r_arr = np.asarray(r_log); th_arr = np.asarray(th_log)
    near_mask = (r_arr <= NEAR_R + 1e-9)
    if np.any(near_mask):
        th_unwrap = unwrap_angle_seq(th_arr[near_mask])
        ω = np.diff(th_unwrap, prepend=th_unwrap[:1]) / DT_SIM
        omega_near_mean = float(np.mean(np.abs(ω)))
        omega_near_peak = float(np.max(np.abs(ω)))
    else:
        omega_near_mean = 0.0; omega_near_peak = 0.0

    diag = dict(
        final_r=float(final_r), min_r=float(min_r),
        area_m2=float(area_m2), fill01=float(fill01),
        path_len=float(path_len), cov_score=float(cov_score),
        roll1_done=None,
        omega_near_mean=omega_near_mean, omega_near_peak=omega_near_peak
    )
    return diag, np.array(ctrl_applied_log, float)

# -------- RIGHT-only optimization cost --------
def cost_right_only(model, ee_sid, goal_xyz, coeffs):
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)

    # place cylinder and obstacle
    init_cylinder_pose(model, d, goal_xyz)
    obs_xyz = compute_obstacle_pose(goal_xyz)
    set_obstacle_pose_mocap(model, d, obs_xyz, cylinder_face_on_quat())

    try:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance  = min(model.opt.tolerance, 1e-9)

    bids = subchain_ids(model, "B12", "B23")
    seq_u, phase_id = control_trajectory_from_coeffs(
        HORIZON_T, CTRL_HZ, coeffs, dir_sign=+1.0
    )
    ctrl_dt   = 1.0/CTRL_HZ
    steps_per = max(1, int(ctrl_dt/DT_SIM))

    nu = model.nu
    p = np.full(nu, INIT_PRESSURE, float); p_smooth = p.copy()

    mujoco.mj_forward(model, d)
    tip = d.site_xpos[ee_sid].copy()
    r0  = float(np.linalg.norm(tip - goal_xyz))
    min_r = r0; final_r = r0; path_len = 0.0

    # roll / grasp state
    back_started   = False
    back_done      = False
    front_started  = False
    front_done     = False
    dwell_steps_left = 0
    roll_back_cum  = 0.0
    roll_front_cum = 0.0
    th_prev_unwrap = None
    grasp_started  = False
    grasp_step_idx = 0

    roller_idx_back  = 0
    roller_idx_front = 1

    for k in range(seq_u.shape[0]):
        u_k = seq_u[k].copy()
        for _ in range(steps_per):
            r_now = float(np.linalg.norm(tip - goal_xyz))

            if r_now > NEAR_R + 0.002 and phase_id[k] == 3:
                uc = 0.5*(u_k[0] + u_k[1])
                ud = 0.5*(u_k[0] - u_k[1])
                gate = smooth_gate((NEAR_R + ROLL_UNLOCK_MARGIN) - r_now, ROLL_UNLOCK_WIDTH)
                ud *= gate
                if r_now > NEAR_R:
                    ud = 0.0
                uc = np.clip(uc * APPROACH_GAIN_CENTER, 0.0, 1.0)
                ud = np.clip(ud * APPROACH_GAIN_DIFF,  -0.5, 0.5)
                u_eff = np.array([uc + ud, uc - ud], dtype=float)
                slew  = PRESSURE_SLEW * APPROACH_SLEW_MULT
                p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM,
                                                  slew_override=slew)

            elif r_now <= NEAR_R + 0.004:
                _, th_now = polar_of(tip, goal_xyz)
                if th_prev_unwrap is None:
                    th_prev_unwrap = th_now

                dth_raw = (th_now - th_prev_unwrap + math.pi) % (2*math.pi) - math.pi
                dth     = clamp_ang_step(dth_raw)
                th_prev_unwrap = th_prev_unwrap + dth

                if not back_started and not front_started:
                    back_started = True
                    roll_back_cum  = 0.0
                    roll_front_cum = 0.0
                    front_done     = False
                    grasp_started  = False
                    grasp_step_idx = 0
                    dwell_steps_left = 0

                if back_started and not back_done:
                    roll_back_cum += max(0.0, abs(dth))
                    progressB = float(np.clip(
                        roll_back_cum / max(BACKROLL_TARGET_RAD, 1e-6), 0.0, 1.0
                    ))
                    u_state = first_roll_override(progressB, roller_idx_back)
                    if roll_back_cum >= BACKROLL_TARGET_RAD:
                        back_done      = True
                        front_started  = True
                        roll_front_cum = 0.0

                elif front_started and not front_done:
                    roll_front_cum += max(0.0, abs(dth))
                    progressF = float(np.clip(
                        roll_front_cum / max(FRONTROLL_TARGET_RAD, 1e-6), 0.0, 1.0
                    ))
                    u_state = first_roll_override(progressF, roller_idx_front)
                    if roll_front_cum >= FRONTROLL_TARGET_RAD:
                        front_done      = True
                        dwell_steps_left = max(1, int(ROLL_DWELL_SEC / DT_SIM))

                elif front_done and dwell_steps_left > 0 and r_now > HOLD_TARGET_R:
                    dwell_steps_left -= 1
                    u_state = post_roll_glide_control(r_now, roller_idx_front)

                else:
                    if r_now > HOLD_TARGET_R:
                        u_state = post_roll_glide_control(r_now, roller_idx_front)
                    else:
                        if not grasp_started:
                            grasp_started  = True
                            grasp_step_idx = 0
                        u_state = perfect_grasp_control(roller_idx_front, grasp_step_idx)
                        grasp_step_idx += 1

                # S-shaped radial blending
                lin = np.clip(
                    (ROLL_NEAR_R - r_now) / max(ROLL_NEAR_R - NEAR_R, 1e-6),
                    0.0, 1.0
                )
                w_r = 0.5 * (1.0 + np.tanh(3.0 * (lin - 0.5)))
                u_eff = blend_u(u_k, u_state, w_r)
                p, p_smooth = apply_pressure_step(d, u_eff, p, p_smooth, DT_SIM)

            else:
                p, p_smooth = apply_pressure_step(d, u_k, p, p_smooth, DT_SIM)

            last_tip = d.site_xpos[ee_sid].copy()
            mujoco.mj_step(model, d)
            tip = d.site_xpos[ee_sid].copy()
            r = float(np.linalg.norm(tip - goal_xyz))
            min_r = min(min_r, r); final_r = r
            path_len += float(np.linalg.norm(tip - last_tip))

    cov_score, _ = coverage_score(d, bids, goal_xyz)
    area_m2, fill01 = area_fill_score(d, bids, goal_xyz, rays=24)

    STALL_PATH = 0.05
    stall_pen  = max(STALL_PATH - path_len, 0.0)

    final_r_now = float(np.linalg.norm(d.site_xpos[ee_sid] - goal_xyz))
    miss       = max(final_r_now - HOLD_TARGET_R, 0.0)
    near_term  = max(min_r - HOLD_TARGET_R, 0.0)

    amp_reg  = 0.0
    warp_reg = 0.0
    env_reg  = 0.0

    cost = (2.0*miss + 1.0*near_term
            - 1.6*cov_score - 0.55*area_m2 - 0.35*fill01
            + 0.8*stall_pen + amp_reg + warp_reg + env_reg)
    return cost

# -------- CMA optimize (RIGHT only) --------
_g_model = None
_g_ee_sid = None

def _worker_init():
    global _g_model, _g_ee_sid
    _g_model  = mujoco.MjModel.from_xml_path(XML_PATH)
    _g_model.opt.timestep = DT_SIM
    try:
        _g_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass
    _g_model.opt.iterations = max(_g_model.opt.iterations, 260)
    _g_model.opt.tolerance  = min(_g_model.opt.tolerance, 1e-9)
    _g_ee_sid  = site_id(_g_model, EE_SITE_NAME, default_last=True)

def _worker_eval(args):
    x, goal = args
    return cost_right_only(_g_model, _g_ee_sid, goal, x)

def default_coeffs():
    b0  = 0.40
    bc  = np.array([+0.08, +0.00])
    bs  = np.array([+0.00, +0.00])
    s0  = 0.12
    sc  = np.array([+0.14, -0.04])
    ss  = np.array([+0.00, +0.00])
    φ   = 0.0
    kappa_raw = 0.0; eps_raw = 0.0
    warp_k    = -0.8; warp_c = 0.60
    s1_raw, s2_raw, s3_raw, s4_raw, w_raw = -0.3, +0.3, +0.7, +1.1, -0.2
    aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw = 0.0, 0.4, 0.2, 0.5, 0.3
    return pack_coeffs(
        b0, bc, bs, s0, sc, ss, φ, kappa_raw, eps_raw, warp_k, warp_c,
        s1_raw, s2_raw, s3_raw, s4_raw, w_raw,
        aF1_raw, aB1_raw, aF2_raw, aB2_raw, aF3_raw
    )

def seed_bank():
    seeds = []
    seeds.append(pack_coeffs(
        0.36, np.array([+0.10, 0.00]), np.array([0.00,0.00]),
        0.10, np.array([+0.20,-0.05]), np.array([0.00,0.00]),
        0.0, +0.2, +0.1, -1.0, 0.58,
        -0.8, -0.2, +0.3, +0.9, 0.0,
        +0.2, +0.6, +0.3, +0.6, +0.5
    ))
    seeds.append(pack_coeffs(
        0.34, np.array([+0.08, +0.04]), np.array([0.00,0.00]),
        0.10, np.array([+0.16, +0.10]), np.array([0.00,0.00]),
        0.0, 0.0, 0.0, -0.6, 0.55,
        -0.4, +0.2, +0.8, +1.2, -0.1,
        +0.3, +0.8, +0.2, +0.9, +0.4
    ))
    seeds.append(pack_coeffs(
        0.48, np.array([+0.04, +0.14]), np.array([0.00,0.00]),
        0.12, np.array([+0.14,  0.00]), np.array([0.00,0.00]),
        0.0, +0.1, +0.0, -0.5, 0.68,
        -0.2, +0.5, +1.0, +1.3, +0.2,
        +0.4, +0.7, +0.6, +0.7, +0.7
    ))
    return seeds

def optimize_right(model, ee_sid, goal_right, warm=None,
                   iters=23, sigma=0.35, pop=None, n_workers=None):
    dim = (1+2*N_HARM) + (1+2*N_HARM) + 5 + 5
    x0  = default_coeffs() if warm is None else warm.copy()
    if pop is None:
        pop = max(60, 3*dim)
    es = CMAEvolutionStrategy(x0, sigma, {'popsize': pop})
    seeds = seed_bank()

    if n_workers is None:
        n_workers = max(1, min(max(1, os.cpu_count() // 2), pop))

    pool = None
    if n_workers > 1:
        ctx  = mp.get_context("spawn")
        pool = ctx.Pool(processes=n_workers, initializer=_worker_init)

    best_cost = 1e9
    best_x    = x0.copy()

    for it in range(iters):
        X = es.ask()

        # seed injection
        for i, s in enumerate(seeds):
            if i < len(X):
                X[i] = s + np.random.normal(0, 0.05, size=x0.shape)

        if pool is None:
            F = []
            for x in X:
                c = cost_right_only(model, ee_sid, goal_right, x)
                F.append(c)
                if c < best_cost:
                    best_cost, best_x = c, x.copy()
        else:
            args = [(x, goal_right) for x in X]
            F = pool.map(_worker_eval, args)
            for i, c in enumerate(F):
                if c < best_cost:
                    best_cost, best_x = c, X[i].copy()

        es.tell(X, F)
        if (it % 2) == 0:
            print(f"[CMA] it {it+1}/{iters} cost={best_cost:.4f}")

    if pool is not None:
        pool.close()
        pool.join()

    return best_x, best_cost

# ===== Randomization (same as before) =====
class RandomPairMirrorSampler:
    """
    Shuffles exactly over the provided GOAL_XZ coordinates.
    Each draw returns (|x|, z) so the caller can form right/left pairs.
    """
    def __init__(self, goals_in_order, seed=SEED):
        self._rng = random.Random(seed + 9991)
        self.items = [(float(x), float(z)) for (x, z) in goals_in_order]
        self._rng.shuffle(self.items)
        self.i = 0
    def next(self):
        if self.i >= len(self.items):
            self._rng.shuffle(self.items)
            self.i = 0
        x, z = self.items[self.i]
        self.i += 1
        return abs(x), z

# ============================================
# -------- Main --------
def main():
    modelA = mujoco.MjModel.from_xml_path(XML_PATH); modelA.opt.timestep = DT_SIM
    modelB = mujoco.MjModel.from_xml_path(XML_PATH); modelB.opt.timestep = DT_SIM

    eeA   = site_id(modelA, EE_SITE_NAME, default_last=True)
    eeB   = site_id(modelB, EE_SITE_NAME, default_last=True)

    try:
        modelA.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        modelB.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    except Exception:
        pass

    print(f"Loaded: nq={modelA.nq}, nv={modelA.nv}, nu={modelA.nu}, dt={modelA.opt.timestep:.4f}")
    print(f"EE site: {EE_SITE_NAME} | Cylinder freejoint: {CYL_FREEJOINT_NAME} | Wrap target: {ANG_TARGET_RAD:.3f} rad")

    EPISODES = len(GOAL_XZ)

    warm = None
    sampler = RandomPairMirrorSampler(GOAL_XZ, seed=SEED)
    ab_rows = []

    for ep in range(EPISODES):
        ax, gz = sampler.next()
        ax, gz = float(ax), float(gz)
        goal_right = np.array([+ax, 0.0, gz], float)
        goal_left  = np.array([-ax, 0.0, gz], float)
        print(f"\n=== Pair {ep+1}/{EPISODES} | |x|={ax:.4f}, z={gz:.4f} ===")

        dim = (1+2*N_HARM) + (1+2*N_HARM) + 5 + 5
        pop = max(60, 3*dim)
        n_workers = max(1, os.cpu_count()//2)

        coeffs_best, best_cost = optimize_right(
            modelA, eeA, goal_right,
            warm=warm, iters=23, sigma=0.35,
            pop=pop, n_workers=n_workers
        )

        # RIGHT: show & record
        diagR, u_log, ctrlR = rollout_right_and_record(
            modelA, eeA, goal_right, coeffs_best, show=True
        )
        print(f"[RIGHT] final_r={diagR['final_r']:.4f} min_r={diagR['min_r']:.4f} "
              f"ω_near_mean={diagR['omega_near_mean']:.3f} ω_near_peak={diagR['omega_near_peak']:.3f} "
              f"area={diagR['area_m2']:.6f} fill={diagR['fill01']:.3f} path={diagR['path_len']:.3f} "
              f"roll_seq_done={diagR['roll1_done']}")

        # LEFT: strict mirrored replay (from u_log), also record applied controls
        diagL, ctrlL = replay_left_with_u_log(
            modelB, eeB, goal_left, u_log, show=True
        )
        print(f"[LEFT ] final_r={diagL['final_r']:.4f} min_r={diagL['min_r']:.4f} "
              f"ω_near_mean={diagL['omega_near_mean']:.3f} ω_near_peak={diagL['omega_near_peak']:.3f} "
              f"area={diagL['area_m2']:.6f} fill={diagL['fill01']:.3f} path={diagL['path_len']:.3f}")

        # Save coeffs + EXACT applied control sequences
        np.save(os.path.join(SAVE_DIR, f"ab_pair_ep{ep+1}.npy"), coeffs_best)
        np.save(os.path.join(SAVE_DIR, f"episode_ep{ep+1}_RIGHT_ctrl.npy"), ctrlR)
        np.save(os.path.join(SAVE_DIR, f"episode_ep{ep+1}_LEFT_ctrl.npy"),  ctrlL)

        warm = coeffs_best.copy()

        # Log in A/B row + artifacts
        ab_rows.append({
            "abs_x": float(ax), "z": float(gz),
            "RIGHT": {
                **diagR,
                "goal": [float(+ax), float(gz)], "cost": float(best_cost),
                "ctrl_file": f"episode_ep{ep+1}_RIGHT_ctrl.npy",
                "n_steps": int(ctrlR.shape[0])
            },
            "LEFT":  {
                **diagL,
                "goal": [float(-ax), float(gz)],
                "ctrl_file": f"episode_ep{ep+1}_LEFT_ctrl.npy",
                "n_steps": int(ctrlL.shape[0])
            }
        })

    # Save summary
    with open(os.path.join(SAVE_DIR, "full_mirror_AB_summary.json"), "w") as f:
        json.dump({
            "HORIZON_T": HORIZON_T, "CTRL_HZ": CTRL_HZ, "DT_SIM": DT_SIM,
            "NEAR_R": NEAR_R, "HOLD_TARGET_R": HOLD_TARGET_R,
            "PLAYBACK_SLOWDOWN": PLAYBACK_SLOWDOWN,
            "first_roll": {
                "back_target_rad": BACKROLL_TARGET_RAD,
                "front_target_rad": FRONTROLL_TARGET_RAD,
                "dwell_sec": ROLL_DWELL_SEC
            },
            "episodes_AB": ab_rows
        }, f, indent=2)

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        main()
    except KeyboardInterrupt:
        pass

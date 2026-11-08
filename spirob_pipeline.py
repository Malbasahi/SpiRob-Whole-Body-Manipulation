#!/usr/bin/env python3
"""Tendon trajectory optimization, neural initialization, and grasp execution.

Place this file beside SpiRob.xml and its model/ assets.
Dependencies: numpy scipy mujoco cma scikit-learn

  python test.py optimize --goal 0.1931 0.6790 --show
  python test.py collect --out experts.npz
  python test.py train --data experts.npz --out warm_start.npz
  python test.py refine --weights warm_start.npz --goal 0.1963 0.6327 --show
  python test.py replay --episode episode.npz --show
  python test.py self-test
"""
import os
# Limit worker-local numerical threads before loading NumPy.
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable,'1')
os.environ.setdefault('MUJOCO_GL','glfw')
import time, math, json
from pathlib import Path
import multiprocessing as mp
import numpy as np
import mujoco
from cma import CMAEvolutionStrategy
try:
    import mujoco.viewer as mjv
    HAVE_MJV = True
except ImportError:
    HAVE_MJV = False
XML_PATH = str(Path(__file__).resolve().with_name('SpiRob.xml'))
EE_SITE_NAME = 's23_3'
CYL_FREEJOINT_NAME = 'O1_freejoint'
SEED = 123
DT_SIM = .002
PLAYBACK_SLOWDOWN = 1.0

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

def smooth_steps(τ, s1, s2, s3, s4, w):
    def S(x): return 0.5*(1+np.tanh(x/(w+1e-6)))
    σ1 = 1 - S(τ - s1)
    σ2 = S(τ - s1) * (1 - S(τ - s2))
    σ3 = S(τ - s2) * (1 - S(τ - s3))
    σ4 = S(τ - s3) * (1 - S(τ - s4))
    σ5 = S(τ - s4)
    Z  = np.clip(σ1 + σ2 + σ3 + σ4 + σ5, 1e-6, None)
    return σ1/Z, σ2/Z, σ3/Z, σ4/Z, σ5/Z

def site_id(model, name, default_last=False):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0 and default_last:
        return model.nsite - 1
    if sid < 0:
        raise RuntimeError(f"Site not found: {name}")
    return sid

def init_cylinder_pose(model, data, goal_xyz):
    """
    Place cylinder body O1 at goal_xyz using its free joint O1_freejoint,
    keeping the default orientation from qpos0.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, CYL_FREEJOINT_NAME)
    if jid < 0:
        raise RuntimeError(f"Freejoint '{CYL_FREEJOINT_NAME}' not found in model.")
    adr = model.jnt_qposadr[jid]

    # Data must be reset to qpos0 before object placement.
    quat0 = data.qpos[adr+3:adr+7].copy()
    data.qpos[adr:adr+3] = np.asarray(goal_xyz, float)
    data.qpos[adr+3:adr+7] = quat0
    mujoco.mj_forward(model, data)

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

from dataclasses import dataclass, asdict
import argparse
import hashlib

METHOD_ID = "spirob-full-wrap-polynomial24-v4"
METHOD_NOTES = """
The 24 coefficients describe shared/differential quartic signals, a sigmoid
warp, phase durations, and five amplitude adjustments. Positive phase durations
reserve time for retreat, reversal, approach, wrapping, and hold. The retreat
uses maximum tendon actuation with the opposing tendon released. Extra retreat
time allows the curl to develop before reversal. Forward wrapping retains
strong rolling actuation and low support tension before final equalization. Phase signs
are constrained to retreat first and then return toward the object. A smooth
start, actuator lag, and phase-specific slew limits bound applied commands.
For objects at negative x the command channels are swapped. --backward-channel
selects the retreat tendon for objects at positive x.

Geometry is measured relative to the moving object. The grasp cost combines
bounded angular progression and body proximity with slow approach, actual
retreat, terminal body coverage, contact retention, and low terminal object
speed. Contact retention is a geometric diagnostic, not proof of force closure.
Throwing additionally scores the final contact-to-separation transition.

The neural network predicts 24 trajectory coefficients from six task inputs.
Predictions initialize CMA-ES and are evaluated by simulation before execution.
Saved data, network weights, and episodes carry model and controller metadata.
"""


@dataclass(frozen=True)
class ControlSettings:
    horizon: float = 10.6
    backward_hold: float = 2.0
    wrap_hold: float = 2.0
    closure_hold: float = .6
    dt: float = 0.002
    ctrl_hz: float = 220.0
    wwrap: float = 1.0
    wgap: float = 10.0
    wdir: float = 1.0
    wvel: float = 1.0
    failure: float = 1000.0
    body_start: int = 12
    body_end: int = 23
    backward_channel: int = 0
    pressure_tau: float = .12
    retreat_slew: float = .65
    approach_slew: float = .30
    closure_slew: float = .45
    approach_speed: float = .08
    retreat_distance: float = .025
    hold_seconds: float = .6
    wmotion: float = 3.
    wspeed: float = 4.
    wcontact: float = 8.
    wcoverage: float = 4.
    wobject: float = 2.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Settings must be finite")
        if self.horizon <= 0 or self.dt <= 0 or self.ctrl_hz <= 0:
            raise ValueError("Time settings must be positive")
        if abs(self.horizon / self.dt - round(self.horizon / self.dt)) > 1e-8:
            raise ValueError("Horizon must be an integer number of simulation steps")
        if min(self.wwrap, self.wgap, self.wdir, self.wvel, self.failure) <= 0:
            raise ValueError("Objective weights and failure penalty must be positive")
        if self.backward_channel not in (0, 1):
            raise ValueError("Backward channel must be 0 or 1")
        if min(self.pressure_tau,self.retreat_slew,self.approach_slew,self.closure_slew,
               self.approach_speed,self.retreat_distance,self.hold_seconds) <= 0:
            raise ValueError("Motion limits must be positive")
        if min(self.wmotion,self.wspeed,self.wcontact,self.wcoverage,self.wobject) < 0:
            raise ValueError("Motion cost weights must be nonnegative")
        if min(self.backward_hold,self.wrap_hold,self.closure_hold) < 0 or self.horizon-self.backward_hold-self.wrap_hold-self.closure_hold < 3.5:
            raise ValueError("Holds must be nonnegative and leave at least 3.5 seconds for the motion schedule")
        if not 0 <= self.body_start <= self.body_end <= 23:
            raise ValueError("Body samples must be a nonempty subset of B0..B23")


def build_condition(x, z, task="grasp", direction=(1., 0.), vmin=0.):
    """Encode object position, task flag, direction, and minimum throw speed."""
    xi = np.array([x, z, task == "throw", 0., 0., 0.], float)
    if task == "throw":
        direction = np.asarray(direction, float)
        if direction.shape != (2,) or np.linalg.norm(direction) < 1e-12 or vmin <= 0:
            raise ValueError("Throw requires a nonzero x-z direction and positive vmin")
        xi[3:5] = direction / np.linalg.norm(direction)
        xi[5] = vmin
    return validate_condition(xi)


def validate_condition(xi):
    xi = np.asarray(xi, float)
    if xi.shape != (6,) or not np.isfinite(xi).all() or xi[2] not in (0., 1.):
        raise ValueError("Condition must be six finite values with task=0 or 1")
    if xi[2] == 0 and np.any(xi[3:] != 0):
        raise ValueError("Grasp conditions require zero direction and speed entries")
    if xi[2] == 1 and (not np.isclose(np.linalg.norm(xi[3:5]), 1.) or xi[5] <= 0):
        raise ValueError("Throw conditions require unit direction and positive speed")
    return xi


def default_params():
    """Initialize approach and closure coefficients for a full-curl retreat."""
    return np.array([.4, 0, 0, 0, 0,
                     1., 0, 0, 0, 0,
                     .5, 0., -.8, .6,
                     0., 0., 0., 0., -.4,
                     .8, .65, .7, .6, .5], float)


def phase_durations(p):
    """Reserve physical time for every motion stage; optimize the remainder."""
    from scipy.special import softmax
    minimum = np.array([.20, .08, .30, .20, .16])
    return minimum + .06*softmax(np.r_[np.asarray(p)[14:18],0.])


def phase_end_times(p, cfg):
    """Allocate additional dwell time to full retreat and forward wrapping."""
    ends = np.cumsum(phase_durations(p))*(cfg.horizon-cfg.backward_hold-cfg.wrap_hold-cfg.closure_hold)+cfg.backward_hold
    ends[3:] += cfg.wrap_hold
    ends[4] += cfg.closure_hold
    return ends


def commands(p, times, horizon=10.6, side=1., backward_channel=0, backward_hold=2., wrap_hold=2., closure_hold=.6):
    """Generate time-indexed targets with a fixed retreat-to-approach order."""
    from scipy.special import expit
    p = np.asarray(p, float)
    if p.shape != (24,) or not np.isfinite(p).all():
        raise ValueError("Trajectory parameters must contain exactly 24 finite values")
    base_horizon = horizon-backward_hold-wrap_hold-closure_hold
    if base_horizon <= 0 or min(backward_hold,wrap_hold,closure_hold) < 0:
        raise ValueError("Horizon must exceed the combined nonnegative retreat, wrap, and closure holds")
    seconds = np.asarray(times,float)
    retreat_end = base_horizon*phase_durations(p)[0]
    extended_retreat_end = retreat_end+backward_hold
    # Stretch only the retreat clock; subsequent phase durations remain unchanged.
    effective_seconds = np.where(seconds < extended_retreat_end,
                                  seconds*retreat_end/extended_retreat_end,
                                  seconds-backward_hold)
    # Extend wrapping without changing retreat, approach, or final hold durations.
    base_ends = np.cumsum(phase_durations(p))*base_horizon
    wrap_start = base_ends[2]+backward_hold
    wrap_end = base_ends[3]+backward_hold+wrap_hold
    wrap_seconds = base_ends[2]+(seconds-wrap_start)*(base_ends[3]-base_ends[2])/(wrap_end-wrap_start)
    effective_seconds = np.where(seconds < wrap_start,effective_seconds,
                                 np.where(seconds < wrap_end,wrap_seconds,seconds-backward_hold-wrap_hold))
    close_duration = base_horizon-base_ends[3]
    closing_seconds = base_ends[3]+(seconds-wrap_end)*close_duration/(close_duration+closure_hold)
    effective_seconds = np.where(seconds < wrap_end,effective_seconds,closing_seconds)
    t = np.clip(effective_seconds/base_horizon,0.,1.)
    if not np.isfinite(t).all():
        raise ValueError("Times must be finite")
    kw,cw = np.clip(p[12],-6.,6.),np.clip(p[13],0.,1.)
    lo,hi = expit(-kw*cw),expit(kw*(1.-cw))
    zeta = t if abs(hi-lo)<1e-12 else (expit(kw*(t-cw))-lo)/(hi-lo)
    eta = zeta-.5
    powers = np.stack([eta**k/math.factorial(k) for k in range(1,5)],axis=-1)
    b = p[0]+powers @ p[1:5]
    s = p[5]+powers @ p[6:10]
    transitions = np.cumsum(phase_durations(p))[:4]
    width = .018+.012*expit(p[18])
    windows = np.stack(smooth_steps(t,*transitions,width),axis=-1)
    # Modest phase-level adjustments preserve the direction and pressure envelope.
    amplitude = .8+.2*np.tanh(p[19:24])
    strength = (.25+.08*np.tanh(p[10]-.5))*(1.+.15*np.tanh(s-1.))
    phase_diff = amplitude*np.array([1.,-.35,-1.,-.45,0.])
    differential = strength*(windows @ phase_diff)
    differential += .015*np.tanh(p[11])*(1.-windows[...,-1])
    center = windows @ np.array([.28,.32,.43,.65,.92])
    center += .07*np.tanh(b-.4)*(1.-windows[...,-1])
    sign = (1. if side>=0 else -1.) * (1. if backward_channel==0 else -1.)
    # A quintic start has zero initial slope and acceleration.
    startup = np.clip(np.asarray(times,float)/.4,0.,1.)
    startup = startup**3*(10.-15.*startup+6.*startup**2)
    target = np.stack([center+sign*differential,center-sign*differential],axis=-1)
    # Maintain forward rolling torque until the wrapping interval is complete.
    wrap_progress = np.clip((seconds-wrap_start)/(wrap_end-wrap_start),0.,1.)
    support = .12+.28*wrap_progress**3
    wrap_target = np.stack([support,np.full_like(support,.98)],axis=-1)
    if sign < 0:
        wrap_target = wrap_target[...,::-1]
    wrap_weight = windows[...,3,None]
    target = (1.-wrap_weight)*target+wrap_weight*wrap_target
    # Saturate the retreat tendon and release the opposing tendon for full curl.
    retreat_target = np.array([1.,0.]) if sign > 0 else np.array([0.,1.])
    # Keep the maximum-curl plateau independent of the stretched retreat clock.
    retreat_weight = (.5*(1.-np.tanh((seconds-extended_retreat_end)/
                                  (width*base_horizon+1e-6))))[...,None]
    target = (1.-retreat_weight)*target+retreat_weight*retreat_target
    return np.clip(.1+startup[...,None]*(target-.1),0.,1.)


def execution_commands(params, xi, cfg):
    """Apply the same lag and rate limits during search and final execution."""
    steps = int(round(cfg.horizon/cfg.dt))
    times = np.arange(steps)*cfg.dt
    sample_times = np.floor(times*cfg.ctrl_hz+1e-10)/cfg.ctrl_hz
    target = commands(params,sample_times,cfg.horizon,xi[0],cfg.backward_channel,cfg.backward_hold,cfg.wrap_hold,cfg.closure_hold)
    applied = np.empty_like(target)
    current = np.full(2,.1)
    boundaries = phase_end_times(params,cfg)
    for k,t in enumerate(times):
        slew = cfg.retreat_slew if t < boundaries[0] else (
            cfg.approach_slew if t < boundaries[3] else cfg.closure_slew)
        delta = cfg.dt/(cfg.pressure_tau+cfg.dt)*(target[k]-current)
        current = np.clip(current+np.clip(delta,-slew*cfg.dt,slew*cfg.dt),0.,1.)
        applied[k] = current
    return applied


def enclosure(tips, centers, final_body_points, cfg):
    """Measure signed angular progression and mean body-to-object distance."""
    relative = np.asarray(tips) - np.asarray(centers)
    angles = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    awrap = float(angles[-1] - angles[0])
    agap = float(np.linalg.norm(np.asarray(final_body_points)-centers[-1], axis=1).mean())
    return -cfg.wwrap*awrap + cfg.wgap*agap, awrap, agap


def final_release(contact):
    """Return [last in-contact index, first separated index], or None."""
    contact = np.asarray(contact, bool)
    indices = np.flatnonzero(contact[:-1] & ~contact[1:])
    return None if not len(indices) else (int(indices[-1]), int(indices[-1]+1))


def score(tips, centers, body_points, velocities, contact, xi, cfg, params=None):
    """Score enclosure, physical motion, contact retention, and release outcome."""
    xi = validate_condition(xi)
    release = final_release(contact) if xi[2] else None
    if xi[2] and release is None:
        return cfg.failure, {"valid_release": False, "reason": "no_contact_to_separation"}
    end = release[0] if release else len(tips)-1
    _, wrap, gap = enclosure(tips[:end+1], centers[:end+1], body_points[end], cfg)
    # Bound sweep reward so repeated fast revolutions cannot dominate enclosure.
    directed_wrap = (1. if xi[0]>=0 else -1.)*wrap
    cost = -cfg.wwrap*np.clip(directed_wrap,0.,2.*math.pi)+cfg.wgap*gap
    info = {"Awrap_rad":wrap,"Agap_m":gap,"Jenc":float(cost),
            "final_robot_object_contact":bool(contact[-1])}
    n = end+1
    tip_velocity = np.diff(tips[:n],axis=0)/cfg.dt
    relative_velocity = np.diff(tips[:n]-centers[:n],axis=0)/cfg.dt
    speed = np.maximum(np.linalg.norm(tip_velocity,axis=1),np.linalg.norm(relative_velocity,axis=1))
    body_distance = np.linalg.norm(body_points[:n]-centers[:n,None,:],axis=2)
    near = np.min(body_distance[1:],axis=1)<.07
    near_speed_penalty = float(np.mean(np.where(near,np.maximum(speed/cfg.approach_speed-1.,0.)**2,0.))) if len(speed) else 0.
    cost += cfg.wspeed*near_speed_penalty
    relative_goal = centers[0]-tips[0]
    goal_direction = relative_goal/max(np.linalg.norm(relative_goal),1e-12)
    p = default_params() if params is None else params
    retreat_end = min(end,max(1,int(phase_end_times(p,cfg)[0]/cfg.dt)))
    retreat = float(max(0.,np.max(-(tips[:retreat_end+1]-tips[0]) @ goal_direction)))
    retreat_penalty = max(1.-retreat/cfg.retreat_distance,0.)**2
    cost += cfg.wmotion*retreat_penalty
    # Measure terminal coverage only from body points close to the cylinder.
    delta = body_points[end]-centers[end]
    radius = np.linalg.norm(delta,axis=1)
    close = radius <= .035
    angles = np.sort(np.mod(np.arctan2(delta[close,1],delta[close,0]),2.*math.pi))
    span = 0. if len(angles)<2 else float(2.*math.pi-np.max(np.diff(np.r_[angles,angles[0]+2.*math.pi])))
    hold_start = max(0,end-int(round(cfg.hold_seconds/cfg.dt))+1)
    retention = float(np.mean(contact[hold_start:end+1]))
    object_speed = np.linalg.norm(velocities[hold_start:end+1],axis=1)
    terminal_object_speed = float(np.mean(object_speed))
    if not xi[2]:
        coverage_penalty = max(1.-span/math.radians(240.),0.)**2
        coverage_penalty += max(1.-int(close.sum())/6.,0.)**2
        contact_penalty = (1.-retention)**2 + float(not contact[-1])
        object_penalty = float(np.mean(np.maximum(object_speed/.04-1.,0.)**2))
        cost += cfg.wcoverage*coverage_penalty+cfg.wcontact*contact_penalty+cfg.wobject*object_penalty
        info.update(coverage_penalty=coverage_penalty,contact_penalty=contact_penalty,
                    object_motion_penalty=object_penalty,
                    retained_enclosure=bool(contact[-1] and retention>=.85 and span>=math.radians(240.)
                                            and close.sum()>=6 and terminal_object_speed<=.04))
    info.update(retreat_displacement_m=retreat,retreat_penalty=retreat_penalty,
                near_speed_penalty=near_speed_penalty,
                peak_near_speed_m_s=float(np.max(speed[near])) if np.any(near) else 0.,
                terminal_coverage_deg=float(np.degrees(span)),terminal_close_body_samples=int(close.sum()),
                hold_contact_fraction=retention,terminal_object_speed_m_s=terminal_object_speed)
    if release:
        velocity = velocities[release[1]]
        speed_rel = float(np.linalg.norm(velocity))
        adir = 1.-float(velocity @ xi[3:5])/(speed_rel+1e-9)
        avel = float(xi[5]-speed_rel)
        cost += cfg.wdir*adir+cfg.wvel*avel
        info.update(valid_release=True,release_index=release[1],release_time_s=release[1]*cfg.dt,
                    release_velocity_xz=velocity.tolist(),release_speed_m_s=speed_rel,Adir=adir,Avel=avel,
                    direction_error_deg=float(np.degrees(np.arccos(np.clip(
                        float(velocity @ xi[3:5])/max(speed_rel,1e-12),-1.,1.)))))
    return float(cost),info


def load_model(path, cfg):
    path = Path(path).expanduser().resolve()
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except ValueError as exc:
        raise ValueError(f"Cannot load {path}: {exc}\nKeep the model/ assets "
                         "beside SpiRob.xml and ensure every referenced mesh is available.") from exc
    model.opt.timestep = cfg.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    model.opt.iterations = max(model.opt.iterations, 260)
    model.opt.tolerance = min(model.opt.tolerance, 1e-9)
    if model.nu != 2:
        raise ValueError("This method expects exactly two tendon actuators")
    # Fail explicitly; never silently use a different tip or object.
    site_id(model, EE_SITE_NAME)
    for name in ["O1"] + [f"B{i}" for i in range(cfg.body_start, cfg.body_end+1)]:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"Required body missing: {name}")
    return model


def rollout(model, xi, params, cfg=ControlSettings(), show=False, commands=None):
    """One rollout implementation shared by cost evaluation and final execution.

    State is observed ONLY for scoring/logging, never to update the commands.
    commands optionally supplies a fixed step-level sequence for exact replay.
    """
    xi = validate_condition(xi)
    steps = int(round(cfg.horizon/cfg.dt))
    if not np.isclose(model.opt.timestep, cfg.dt):
        raise ValueError("Model timestep differs from settings")
    if commands is None:
        # Hold sample k until next 220Hz boundary, without rounding dt ratios.
        commands = execution_commands(params, xi, cfg)
    else:
        commands = np.asarray(commands, float)
        if commands.shape != (steps, 2) or not np.isfinite(commands).all():
            raise ValueError("Replay controls have incorrect length, shape, or finite values")
        if np.any((commands < 0) | (commands > 1)):
            raise ValueError("Replay controls must lie within [0,1]")
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    init_cylinder_pose(model, d, [xi[0], 0., xi[1]])
    tip_id = site_id(model, EE_SITE_NAME)
    oid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "O1")
    robot_bodies = set()
    for i in range(24):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"B{i}")
        if bid >= 0:
            robot_bodies.add(bid)
    # Include any robot child geometry; exclude unrelated scene geometry.
    robot_descendants = set(robot_bodies)
    for bid in range(1, model.nbody):
        if int(model.body_parentid[bid]) in robot_descendants:
            robot_descendants.add(bid)
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"B{i}")
            for i in range(cfg.body_start, cfg.body_end+1)]
    tips = np.empty((steps+1, 2)); centers = tips.copy(); velocities = tips.copy()
    bodies = np.empty((steps+1, len(bids), 2)); contacts = np.zeros(steps+1, bool)
    velocity6 = np.empty(6)

    def observe(k):
        # mj_step leaves position-derived fields at an integrator stage.
        # Recompute at final qpos/qvel so every logged state is synchronized.
        mujoco.mj_forward(model, d)
        tips[k] = d.site_xpos[tip_id, [0, 2]]
        centers[k] = d.xpos[oid, [0, 2]]
        bodies[k] = d.xipos[bids][:, [0, 2]]
        mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, oid, velocity6, 0)
        velocities[k] = velocity6[[3, 5]]
        for c in d.contact[:d.ncon]:
            if c.efc_address < 0:
                continue  # exclude inactive proximity contacts
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if (b1 == oid and b2 in robot_descendants) or (b2 == oid and b1 in robot_descendants):
                contacts[k] = True
                break

    viewer = SafeViewer(model, d, show=show)
    if show and viewer.v is None:
        raise RuntimeError("Viewer could not open; use a desktop session or omit --show")
    try:
        observe(0)
        for k, command in enumerate(commands):
            d.ctrl[:] = command  # NO state-based modifications
            mujoco.mj_step(model, d)
            observe(k+1)
            if not np.isfinite(d.qpos).all() or d.time < (k+.5)*cfg.dt:
                raise RuntimeError("MuJoCo rollout became unstable or reset its time")
            if show:
                viewer.draw()
                time.sleep(cfg.dt*PLAYBACK_SLOWDOWN)
    finally:
        viewer.close()
    cost, metrics = score(tips, centers, bodies, velocities, contacts, xi, cfg, params)
    if not np.isfinite(cost):
        raise RuntimeError("Nonfinite rollout objective")
    metrics.update(cost=cost, duration_s=float(d.time),
                   min_tip_distance_m=float(np.linalg.norm(tips-centers, axis=1).min()),
                   final_tip_distance_m=float(np.linalg.norm(tips[-1]-centers[-1])),
                   contact_samples=int(contacts.sum()))
    return dict(cost=cost, metrics=metrics, condition=xi, params=np.asarray(params),
                ctrl=commands, times=np.arange(steps+1)*cfg.dt,
                tip_xz=tips, object_xz=centers, body_xz=bodies,
                object_velocity_xz=velocities, contact=contacts)


def optimize(model, xi, cfg, warm=None, iters=23, pop=72, sigma=.35,
                   seed=SEED, workers=1, xml_path=None):
    """CMA mean=warm, C0=I. Retain the evaluated incumbent, even if search worsens."""
    x0 = default_params() if warm is None else np.asarray(warm, float).copy()
    commands(x0, [0.], cfg.horizon, backward_hold=cfg.backward_hold, wrap_hold=cfg.wrap_hold, closure_hold=cfg.closure_hold)  # Validate before spawning workers.
    if iters < 1 or pop < 4 or workers < 1 or sigma <= 0:
        raise ValueError("Need iters>=1, pop>=4, workers>=1 and sigma>0")
    best = rollout(model, xi, x0, cfg)
    best_p, best_cost = x0.copy(), best['cost']
    es = CMAEvolutionStrategy(x0, sigma, {"popsize": pop, "seed": seed,
                                          "verbose": -9, "CMA_diagonal": False})
    evaluations = 1
    pool = None
    if workers > 1:
        if xml_path is None:
            raise ValueError("Parallel search needs an XML path")
        pool = mp.get_context("spawn").Pool(min(workers, pop),
                initializer=worker_init, initargs=(str(xml_path), asdict(cfg)))
    try:
        for it in range(iters):
            if es.stop():
                break
            candidates = es.ask()
            if pool:
                values = pool.map(worker_eval, [(p, xi) for p in candidates])
            else:
                values = [rollout(model, xi, p, cfg)['cost'] for p in candidates]
            evaluations += len(values)
            i = int(np.argmin(values))
            if values[i] < best_cost:
                best_cost, best_p = float(values[i]), candidates[i].copy()
            es.tell(candidates, values)
            print(f"[CMA] {it+1}/{iters} cost={best_cost:.6f} rollouts={evaluations}", flush=True)
    finally:
        if pool:
            pool.close(); pool.join()
    return best_p, {"best_cost": best_cost, "initial_cost": best['cost'],
                    "search_rollouts": evaluations, "seed": seed}


def worker_init(xml_path, settings):
    global _model, _cfg
    _cfg = ControlSettings(**settings)
    _model = load_model(xml_path, _cfg)


def worker_eval(item):
    return rollout(_model, item[1], item[0], _cfg)['cost']


def build_metadata(xml_path, cfg):
    # Hash all referenced assets, including meshes in model/, not only XML.
    import xml.etree.ElementTree as ET
    root_path = Path(xml_path).resolve()
    digest = hashlib.sha256()
    visited = set()

    def add_file(path, parse_xml=False):
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        digest.update(path.name.encode()); digest.update(path.read_bytes())
        if parse_xml:
            root = ET.parse(path).getroot()
            compiler = root.find("compiler")
            meshdir = compiler.get("meshdir", "") if compiler is not None else ""
            texturedir = compiler.get("texturedir", "") if compiler is not None else ""
            for elem in root.iter():
                name = elem.get("file")
                if name:
                    directory = meshdir if elem.tag == "mesh" else texturedir if elem.tag == "texture" else ""
                    add_file(path.parent / directory / name, elem.tag == "include")
    add_file(root_path, True)
    return {"method": METHOD_ID, "model_sha256": digest.hexdigest(),
            "settings": asdict(cfg), "mujoco_version": mujoco.__version__,
            "parameter_dim": 24, "condition_dim": 6}


def require_metadata(actual, expected):
    if actual != expected:
        raise ValueError("Saved method/model/settings/MuJoCo version differ from this run. "
                         "Use the same XML/assets, weights, and settings; regenerate if changed.")


def save_npz(path, **data):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Output path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic checkpoints survive interruption without truncating the last file.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **data)
    os.replace(temp, path)


def fit_control_trace(path, cfg):
    """Fit an applied-control trace to initialize the staged trajectory."""
    from scipy.optimize import least_squares
    target = np.load(path, allow_pickle=False)
    if target.ndim != 2 or target.shape[1] != 2 or len(target) < 2:
        raise ValueError("Control traces must be a step-level (N,2) .npy array")
    if not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
        raise ValueError("Applied controls must be finite and in [0,1]")
    # Control-trace samples are spaced by DT_SIM seconds.
    count = min(len(target),int(round(cfg.horizon/cfg.dt)))
    if count < 2:
        raise ValueError("Control trace must contain at least two usable samples")
    fit = least_squares(lambda p: (execution_commands(p,build_condition(1.,.7),cfg)[:count]-target[:count]).ravel(),
                        default_params(), max_nfev=400)
    print(f"Control-trace fit RMSE={np.sqrt(np.mean(fit.fun**2)):.6f}; "
          "this is waveform agreement, not grasp success")
    return fit.x


def train(data_path, out_path, epochs=2000):
    """Supervised standardized neural regression; export NumPy-only inference."""
    from sklearn.neural_network import MLPRegressor
    with np.load(data_path, allow_pickle=False) as src:
        X, Y = src['conditions'], src['params']
        metadata = json.loads(str(src['metadata']))
    if metadata.get('method') != METHOD_ID or X.ndim != 2 or X.shape[1] != 6 or Y.shape != (len(X),24):
        raise ValueError("Dataset method or shape mismatch")
    if len(X) < 4 or not np.isfinite(Y).all():
        raise ValueError("Need at least four finite expert pairs to train")
    for xi in X:
        validate_condition(xi)
    xm, xs = X.mean(axis=0), X.std(axis=0)
    ym, ys = Y.mean(axis=0), Y.std(axis=0)
    xs = np.where(xs < 1e-8, 1., xs); ys = np.where(ys < 1e-8, 1., ys)
    net = MLPRegressor(hidden_layer_sizes=(64,64), activation='tanh', solver='lbfgs',
                       max_iter=epochs, max_fun=max(15000, epochs*10), random_state=SEED,
                       alpha=1e-4, tol=1e-7)
    net.fit((X-xm)/xs, (Y-ym)/ys)
    arrays = {f'W{i}': w for i,w in enumerate(net.coefs_)}
    arrays.update({f'b{i}': b for i,b in enumerate(net.intercepts_)})
    arrays.update(x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys, x_min=X.min(axis=0),
                  x_max=X.max(axis=0),metadata=json.dumps(metadata))
    exported = predict_arrays(arrays, X)
    reference = net.predict((X-xm)/xs)*ys+ym
    if not np.allclose(exported, reference, atol=1e-10, rtol=1e-10):
        raise RuntimeError("Exported NumPy NN differs from fitted network")
    save_npz(out_path, **arrays)
    print(f"NN saved: {out_path}; training parameter RMSE={np.sqrt(np.mean((exported-Y)**2)):.6g}. "
          "This is not held-out manipulation performance.")


def predict_arrays(weights, conditions):
    h = (np.asarray(conditions)-weights['x_mean'])/weights['x_scale']
    for i in range(3):
        h = h @ weights[f'W{i}'] + weights[f'b{i}']
        if i < 2:
            h = np.tanh(h)
    return h*weights['y_scale']+weights['y_mean']


def predict(path, xi, metadata):
    with np.load(path, allow_pickle=False) as w:
        require_metadata(json.loads(str(w['metadata'])), metadata)
        if np.any(xi < w['x_min']-1e-8) or np.any(xi > w['x_max']+1e-8):
            print("Notice: condition is outside an expert-data coordinate range; "
                  "the NN is extrapolating and still requires refinement.")
        return predict_arrays(w, xi)


def self_test():
    """Check phase ordering, rate limits, geometry, retention, and release."""
    cfg = ControlSettings()
    p = default_params(); times = np.arange(round(cfg.horizon/cfg.dt))*cfg.dt
    u = execution_commands(p,build_condition(.1931,.679),cfg)
    assert u.shape == (5300,2) and np.isfinite(u).all()
    assert np.allclose(u[0],[.1,.1]) and u.min()>=0 and u.max()<=1
    limits = np.where(times < phase_end_times(p,cfg)[0],cfg.retreat_slew,
                     np.where(times < phase_end_times(p,cfg)[3],cfg.approach_slew,cfg.closure_slew))
    assert np.all(np.abs(np.diff(np.vstack([[.1,.1],u]),axis=0))<=limits[:,None]*cfg.dt+1e-12)
    retreat = u[(times>.5)&(times<1.)]
    forward = u[(times>4.8)&(times<5.5)]
    full_curl = u[(times>2.)&(times<2.9)]
    assert np.min(full_curl[:,0])>.99 and np.max(full_curl[:,1])<.01
    assert np.isclose(phase_end_times(p,cfg)[0]-6.*phase_durations(p)[0],2.)
    # Extra wrapping time starts after approach and ends before final hold.
    base_ends = np.cumsum(phase_durations(p))*6.+2.
    assert np.allclose(phase_end_times(p,cfg)[:3],base_ends[:3])
    assert np.isclose(phase_end_times(p,cfg)[3],base_ends[3]+2.)
    assert np.isclose(phase_end_times(p,cfg)[4],base_ends[4]+2.6)
    rolling = u[(times>6.6)&(times<7.5)]
    assert np.min(rolling[:,1])>.95 and np.max(rolling[:,0])<.35
    assert np.min(rolling[:,1]-rolling[:,0])>.6
    assert np.all(retreat[:,0]>retreat[:,1]) and np.all(forward[:,1]>forward[:,0])
    opposite = execution_commands(p,build_condition(-.1931,.679),cfg)
    assert np.allclose(opposite,u[:,::-1])
    assert np.allclose(u,execution_commands(p,build_condition(.22,.60),cfg))
    assert np.max(np.abs(u[-1]-u[-1].mean()))<.015
    assert np.min(u[-1])>.85
    # Large optimizer perturbations must preserve direction ordering and bounds.
    rng = np.random.default_rng(7)
    for _ in range(30):
        q = p+rng.normal(0,4,24)
        durations = phase_durations(q)
        assert np.isclose(durations.sum(),1.) and durations[0]>=.2 and durations[2]>=.3
        back = commands(q,[cfg.horizon*.10],cfg.horizon)
        approach = commands(q,[5.],cfg.horizon)
        assert back[0,0]>back[0,1] and approach[0,1]>approach[0,0]
    # A compact stationary enclosure scores better than a fast, contact-free pass.
    n = 1001
    angles = np.linspace(0,2*np.pi,12,endpoint=False)
    centers = np.zeros((n,2)); tips = np.tile([.02,0.],(n,1))
    bodies = np.tile(.02*np.stack([np.cos(angles),np.sin(angles)],axis=1),(n,1,1))
    velocities = np.zeros((n,2)); contact = np.ones(n,bool); xi=build_condition(.1,.7)
    good,diagnostic = score(tips,centers,bodies,velocities,contact,xi,cfg,p)
    bad,_ = score(tips,centers,bodies,velocities,~contact,xi,cfg,p)
    assert good<bad and diagnostic['retained_enclosure']
    fast = tips.copy();fast[:,0]+=.002*np.sin(np.arange(n)*.5)
    fast_cost,info = score(fast,centers,bodies,velocities,contact,xi,cfg,p)
    assert fast_cost>good and info['near_speed_penalty']>0
    assert final_release([False,True,False,True,False]) == (3,4)
    assert final_release([False,False]) is None
    throw_xi = build_condition(.1,.7,'throw',(1,0),1.)
    assert score(tips,centers,bodies,velocities,~contact,throw_xi,cfg,p)[0] == cfg.failure
    try:
        commands(np.zeros(25),times,cfg.horizon)
    except ValueError:
        pass
    else:
        raise AssertionError('Incorrect parameter count was accepted')
    print('PASS: full-curl actuation and dwell, sustained forward rolling torque, delayed closure, retreat/approach order, phase timing, smooth startup, applied slew limits, '
          'channel symmetry, terminal closure, bounded perturbations, contact retention, '
          'near-object speed cost, release detection, and parameter validation.')


def cli():
    global XML_PATH
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', nargs='?', default='optimize',
                        choices=['optimize','collect','train','refine','replay','self-test','method-notes'])
    parser.add_argument('--xml',default=XML_PATH)
    parser.add_argument('--out', help='Output .npz (default depends on stage)')
    parser.add_argument('--goal',nargs=2,type=float,default=[.1931,.6790],metavar=('X','Z'))
    parser.add_argument('--task',choices=['grasp','throw'],default='grasp')
    parser.add_argument('--direction',nargs=2,type=float,default=[1.,0.],metavar=('DX','DZ'))
    parser.add_argument('--vmin',type=float,default=1.)
    parser.add_argument('--iters',type=int,default=23)
    parser.add_argument('--pop',type=int,default=72)
    parser.add_argument('--sigma',type=float,default=.35)
    parser.add_argument('--workers',type=int,default=max(1,min(8,(os.cpu_count() or 2)//2)))
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--data',default='experts.npz')
    parser.add_argument('--weights',default='warm_start.npz')
    parser.add_argument('--episode',default='episode.npz')
    parser.add_argument('--epochs',type=int,default=2000)
    parser.add_argument('--limit',type=int,default=84,help='Expert conditions to collect (up to 84)')
    parser.add_argument('--resume',action='store_true',help='Resume expert collection checkpoint')
    parser.add_argument('--seed-controls',help='Optional step-level .npy control trace for initialization')
    parser.add_argument('--horizon',type=float,default=10.6,help='Total motion duration in simulation seconds')
    parser.add_argument('--backward-hold',type=float,default=2.,help='Extra seconds allocated to the maximum-curl retreat')
    parser.add_argument('--wrap-hold',type=float,default=2.,help='Extra seconds of forward wrapping before equalization')
    parser.add_argument('--closure-hold',type=float,default=.6,help='Extra seconds for support-tendon tightening to settle')
    parser.add_argument('--approach-slew',type=float,default=.30,help='Maximum forward-phase command change per second')
    parser.add_argument('--approach-speed',type=float,default=.08,help='Desired near-object speed limit in metres per second')
    parser.add_argument('--backward-channel',type=int,choices=[0,1],default=0,help='Retreat tendon for an object at positive x')
    parser.add_argument('--wwrap',type=float,default=1.)
    parser.add_argument('--wgap',type=float,default=10.)
    parser.add_argument('--wdir',type=float,default=1.)
    parser.add_argument('--wvel',type=float,default=1.)
    parser.add_argument('--failure',type=float,default=1000.)
    args = parser.parse_args()
    if args.stage == 'self-test':
        self_test(); return
    if args.stage == 'method-notes':
        print(METHOD_NOTES); return
    if args.stage == 'train':
        train(args.data,args.out or 'warm_start.npz',args.epochs); return
    cfg = ControlSettings(horizon=args.horizon,backward_hold=args.backward_hold,wrap_hold=args.wrap_hold,closure_hold=args.closure_hold,approach_slew=args.approach_slew,
                          approach_speed=args.approach_speed,backward_channel=args.backward_channel,
                          wwrap=args.wwrap,wgap=args.wgap,wdir=args.wdir,wvel=args.wvel,failure=args.failure)
    model = load_model(args.xml,cfg)
    metadata = build_metadata(args.xml,cfg)
    xi = build_condition(*args.goal,args.task,args.direction,args.vmin)
    warm = None
    if args.seed_controls:
        warm = fit_control_trace(args.seed_controls,cfg)
    if args.stage == 'replay':
        with np.load(args.episode,allow_pickle=False) as src:
            require_metadata(json.loads(str(src['metadata'])),metadata)
            result = rollout(model,src['condition'],src['params'],cfg,args.show,commands=src['ctrl'])
            if not np.allclose(result['object_xz'],src['object_xz'],rtol=0,atol=1e-8):
                raise RuntimeError("Replay trajectory differs from saved trajectory")
        print(json.dumps(result['metrics'],indent=2)); return
    if args.stage == 'collect':
        if not 1 <= args.limit <= 84:
            raise ValueError('--limit must be between 1 and 84')
        # Both signs are independently optimized, not assumed mirrored successes.
        goals = [(sign*abs(x),z) for x,z in GOAL_XZ for sign in (1.,-1.)][:args.limit]
        target = args.out or 'experts.npz'
        conditions, params, costs = [], [], []
        if args.resume and Path(target).exists():
            with np.load(target,allow_pickle=False) as src:
                require_metadata(json.loads(str(src['metadata'])),metadata)
                conditions = list(src['conditions']); params = list(src['params']); costs = list(src['costs'])
            expected = np.array([build_condition(*g,args.task,args.direction,args.vmin) for g in goals])
            if len(conditions) > len(goals) or not np.array_equal(np.array(conditions),expected[:len(conditions)]):
                raise ValueError('Resume conditions differ from requested collection')
        for i in range(len(conditions),len(goals)):
            condition = build_condition(*goals[i],args.task,args.direction,args.vmin)
            print(f'Expert {i+1}/{len(goals)}: {condition.tolist()}',flush=True)
            # Initialize search from the most recently completed expert.
            initial = params[-1] if params else warm
            p, report = optimize(model,condition,cfg,initial,args.iters,args.pop,args.sigma,
                                       args.seed+i,args.workers,args.xml)
            conditions.append(condition); params.append(p); costs.append(report['best_cost'])
            save_npz(target,conditions=np.array(conditions),params=np.array(params),costs=np.array(costs),
                     metadata=json.dumps(metadata))
        print(f'Expert dataset saved: {target}. Finite-budget solutions; no success-rate claim.'); return
    if args.stage == 'refine':
        warm = predict(args.weights,xi,metadata)
    p, report = optimize(model,xi,cfg,warm,args.iters,args.pop,args.sigma,
                               args.seed,args.workers,args.xml)
    result = rollout(model,xi,p,cfg,args.show)
    if not np.isclose(result['cost'],report['best_cost'],atol=1e-9,rtol=0):
        raise RuntimeError('Final rollout disagrees with optimization evaluation')
    metrics = {**result.pop('metrics'),**report}
    result.pop('cost')
    save_npz(args.out or 'episode.npz',**result,metrics=json.dumps(metrics),metadata=json.dumps(metadata))
    print(json.dumps(metrics,indent=2))
    print(f"Saved {args.out or 'episode.npz'}")


if __name__ == '__main__':
    mp.freeze_support()
    try:
        cli()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
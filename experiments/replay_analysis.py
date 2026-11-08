#randomized grasping

#!/usr/bin/env python3
# replay_plot_pack_from_saved_ctrl.py
#
# PAPER PLOT PACK via REPLAY ONLY (no CMA rerun).
#
# Replays saved applied controls and exports a "plot pack" directory per episode/side:
#   - r(t) (tip-object distance) vs time
#   - phi(t), phi_unwrap(t), Theta_cum(t)
#   - Cov(t), Fill(t)
#   - Contacts: I_contact(t), n_contact(t), Sum normal force
#   - Jam/slip proxy: I_near(t) overlaid with |dphi|
#   - Cost component accumulations + final cost decomposition bars
#   - Polar occupancy visualization (final)
#   - Snapshots (start / rollback moment / final)
#
# Optional: replay-time "no-rollback" ablation variant (NOT "best no-rollback"):
#   --ablate-no-rollback
#   clamps segments with negative dphi (phi decreasing) by holding ctrl constant
#   so angular rollback is suppressed without rerunning optimization.
#
# Assumptions (adjust via CLI flags):
#   - XML_PATH points to your MuJoCo model
#   - object body is "O1"
#   - object freejoint name "O1_freejoint" for placement
#   - tip site name defaults to "s23_3" (your end-effector site from your generator)
#   - enclosure points use bodies B12..B23 (as in your generator)
#
# Outputs under: SAVE_DIR/plot_pack/episode_ep{N}_{SIDE}/...
#
# Usage:
#   python3 randomized_grasping_camera_video.py
#   python3 randomized_grasping_camera_video.py --record-video
#   python3 randomized_grasping_camera_video.py --no-view --record-video
#   python3 randomized_grasping_camera_video.py --ablate-no-rollback
#
import os
from pathlib import Path
os.environ["MUJOCO_GL"] = "glfw"

import json
import time
import math
import argparse
import csv
from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ---------------- USER PATHS ----------------
XML_PATH           = str(Path(__file__).resolve().parents[1] / "SpiRob.xml")
SAVE_DIR           = str(Path(__file__).resolve().parents[1] / "outputs" / "obstacles")
CYL_FREEJOINT_NAME = "O1_freejoint"
OBJ_BODY_NAME      = "O1"

# ---------------- Defaults aligned to your pipeline ----------------
DEFAULT_TIP_SITE   = "s23_3"                     # EE site used in your generator
DEFAULT_CHAIN      = [f"B{i}" for i in range(12, 24)]  # B12..B23

TENDON_NAMES = ("t1", "t2")

# Camera pose and video capture copied from pick-and-throw.
CAMERA_NAME = "camera1"
PICK_THROW_CAMERA_POS = np.array([0.0, -1.2, 0.75], dtype=float)
PICK_THROW_CAMERA_EULER_X = 1.5708
PICK_THROW_CAMERA_FOVY = 45.0

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 60
POST_VIEW_SEC = 3.0

VIDEO_BRIGHTNESS_GAIN = 1.18
VIDEO_BRIGHTNESS_BIAS = 0.08
DUPLICATE_FRAMES = 2

# Viewer optional
try:
    import mujoco.viewer as mjv
    HAVE_MJV = True
except Exception:
    HAVE_MJV = False

# Offscreen renderer for snapshots (preferred)
HAVE_RENDERER = hasattr(mujoco, "Renderer")


# ---------------- Utilities ----------------
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _load_summary(save_dir):
    names = [
        "full_mirror_AB_summary.json",
        "full_merge_mirror_AB_summary.json",
        "mirror_AB_summary.json",
        "full2_mixed_AB_summary.json",
        "multi_episode_summary.json",
        "single_openloop_summary.json",
    ]
    for n in names:
        path = os.path.join(save_dir, n)
        if os.path.isfile(path):
            with open(path, "r") as f:
                print(f"[info] Using summary: {path}")
                return json.load(f), path
    raise FileNotFoundError(f"No summary JSON found in {save_dir} (tried common names).")

def _mj_name2id_safe(model, objtype, name):
    try:
        i = mujoco.mj_name2id(model, objtype, name)
        return int(i)
    except Exception:
        return -1

def _body_pos(data, body_id):
    # prefer xipos (body inertial frame position) if present
    if hasattr(data, "xipos"):
        return np.asarray(data.xipos[body_id], dtype=float).copy()
    return np.asarray(data.body_xpos[body_id], dtype=float).copy()

def _site_pos(data, site_id):
    return np.asarray(data.site_xpos[site_id], dtype=float).copy()

def _wrap_angle_xz(p_tip, p_obj):
    # angle in x-z plane around object
    dx = float(p_tip[0] - p_obj[0])
    dz = float(p_tip[2] - p_obj[2])
    return math.atan2(dz, dx)

def _unwrap_increment(phi_prev, phi_curr):
    # returns delta phi in (-pi, pi] then unwrapped step
    d = phi_curr - phi_prev
    while d <= -math.pi:
        d += 2.0 * math.pi
    while d > math.pi:
        d -= 2.0 * math.pi
    return d

def _collect_geom_ids_for_body(model, body_id):
    adr = int(model.body_geomadr[body_id])
    num = int(model.body_geomnum[body_id])
    return list(range(adr, adr + num))

def _robot_object_geom_sets(model, obj_body_id):
    obj_geoms = set(_collect_geom_ids_for_body(model, obj_body_id))
    robot_geoms = []
    for gid in range(model.ngeom):
        b = int(model.geom_bodyid[gid])
        if b == 0:
            continue
        if gid in obj_geoms:
            continue
        robot_geoms.append(gid)
    return set(robot_geoms), set(obj_geoms)

def _contact_metrics(model, data, robot_geom_set, obj_geom_set):
    """
    Returns:
      in_contact (bool),
      n_contact_pairs (int),
      sum_normal_force (float)
    """
    in_contact = False
    n_pairs = 0
    sum_fn = 0.0

    # mj_contactForce returns 6D force/torque in contact frame.
    # component 0 is normal force (along contact normal).
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        g1, g2 = int(c.geom1), int(c.geom2)
        ok = ((g1 in robot_geom_set and g2 in obj_geom_set) or
              (g2 in robot_geom_set and g1 in obj_geom_set))
        if not ok:
            continue

        in_contact = True
        n_pairs += 1
        try:
            arr = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(model, data, ci, arr)
            sum_fn += float(arr[0])
        except Exception:
            # if not available, just skip force
            pass

    return bool(in_contact), int(n_pairs), float(sum_fn)

def init_cylinder_pose(model, data, goal_xyz, freejoint_name):
    jid = _mj_name2id_safe(model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name)
    if jid < 0:
        raise RuntimeError(f"Freejoint '{freejoint_name}' not found in model.")
    adr = int(model.jnt_qposadr[jid])

    quat0 = data.qpos[adr+3:adr+7].copy()
    data.qpos[adr:adr+3]   = np.asarray(goal_xyz, float)
    data.qpos[adr+3:adr+7] = quat0
    mujoco.mj_forward(model, data)


# ---------------- Enclosure metrics (replay-time proxies) ----------------
def compute_cov_fill_from_chain_points(chain_pts_xz, obj_xz, n_bins=36,
                                      r_min=0.0, r_max=0.08):
    """
    Simple, robust enclosure proxies from chain points around object:

    - Coverage: fraction of angular bins that have >=1 point within [r_min, r_max]
    - Fill: mean occupancy across bins (clipped), i.e., average count per bin normalized.

    These are not "ground truth" coverage, but they are stable and paper-friendly.
    If you want your exact generator scores, plug those functions in here.
    """
    if chain_pts_xz is None or len(chain_pts_xz) == 0:
        return 0.0, 0.0, np.zeros(n_bins, dtype=int)

    pts = np.asarray(chain_pts_xz, dtype=float).reshape(-1, 2)
    ox, oz = float(obj_xz[0]), float(obj_xz[1])
    dx = pts[:, 0] - ox
    dz = pts[:, 1] - oz
    rr = np.sqrt(dx*dx + dz*dz)

    # keep points in annulus band
    mask = (rr >= float(r_min)) & (rr <= float(r_max))
    if not np.any(mask):
        return 0.0, 0.0, np.zeros(n_bins, dtype=int)

    ang = np.arctan2(dz[mask], dx[mask])  # [-pi, pi]
    # bin to [0, n_bins)
    bin_idx = ((ang + np.pi) / (2.0*np.pi) * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins-1)

    counts = np.bincount(bin_idx, minlength=n_bins).astype(int)
    coverage = float(np.mean(counts > 0))

    # fill proxy: average occupancy, clipped to 1 by a soft cap (keeps it in [0,1])
    # cap at 3 points/bin for saturation
    fill = float(np.mean(np.clip(counts / 3.0, 0.0, 1.0)))
    return coverage, fill, counts


def polar_occupancy_plot(counts, out_png, title="Polar occupancy (final)"):
    counts = np.asarray(counts, dtype=float).reshape(-1)
    n = counts.size
    if n == 0:
        return
    # Polar bar plot
    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    radii = np.clip(counts / max(np.max(counts), 1.0), 0.0, 1.0)

    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(theta, radii, width=(2*np.pi/n), align='edge')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)


# ---------------- Cost definitions (replay-recomputed) ----------------
def update_cost_accumulators(cost, r, theta_cum, cov, fill, u, du,
                             in_contact, near, dphi,
                             w_dist=1.0, r_target=0.0,
                             w_theta=1.0, w_cov=1.0, w_fill=1.0,
                             w_u=0.05, w_du=0.01,
                             w_stall=0.5,
                             stall_dphi_eps=0.002):
    """
    Running accumulations (paper-friendly, consistent across policies).
    You can tune weights later without touching your generator.
    """
    # distance penalty (hinge)
    jd_inc = max(float(r) - float(r_target), 0.0)
    cost["Jd"] += w_dist * jd_inc

    # wrap "reward" (we store as cost component: lower is better)
    # we want more theta => lower cost => subtract theta_cum increment
    # use positive increment only so rollback doesn't "reward" itself
    # NOTE: theta_cum already defined as cumulative positive wrap
    cost["Jtheta"] = -w_theta * float(theta_cum)

    # enclosure as costs (higher cov/fill => lower cost)
    cost["Jcov"]  = -w_cov  * float(cov)
    cost["Jfill"] = -w_fill * float(fill)

    # regularization
    if u is not None:
        cost["Jr"] += w_u * float(np.sum(np.asarray(u, float)**2))
    if du is not None:
        cost["Jr"] += w_du * float(np.sum(np.asarray(du, float)**2))

    # stall penalty: near & contact-ish but no angular progress
    # "jam regime"
    if bool(near) and (bool(in_contact) or True):
        if abs(float(dphi)) < float(stall_dphi_eps):
            cost["Jstall"] += w_stall

    # total (composed)
    cost["Jtotal"] = cost["Jd"] + cost["Jtheta"] + cost["Jcov"] + cost["Jfill"] + cost["Jr"] + cost["Jstall"]
    return cost


# ---------------- Camera, clean viewer, and video helpers ----------------
def configure_pick_throw_camera(model):
    """Apply the pick-and-throw camera pose without changing the XML or physics."""
    cam_id = _mj_name2id_safe(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{CAMERA_NAME}' not found in model.")

    half_angle = 0.5 * PICK_THROW_CAMERA_EULER_X
    model.cam_pos[cam_id] = PICK_THROW_CAMERA_POS
    model.cam_quat[cam_id] = np.array(
        [math.cos(half_angle), math.sin(half_angle), 0.0, 0.0],
        dtype=float,
    )
    model.cam_fovy[cam_id] = PICK_THROW_CAMERA_FOVY
    return cam_id


def launch_clean_passive_viewer(model, data, cam_id):
    """Launch the viewer with both MuJoCo side panels disabled."""
    try:
        viewer_ctx = mjv.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        )
    except TypeError:
        # Compatibility with older MuJoCo releases.
        viewer_ctx = mjv.launch_passive(model, data)

    viewer = viewer_ctx.__enter__()
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = cam_id

    # These attributes are version-dependent; the launch flags above are the
    # primary path, while these assignments cover compatible older handles.
    for attr in ("show_left_ui", "_show_left_ui", "show_right_ui", "_show_right_ui"):
        if hasattr(viewer, attr):
            try:
                setattr(viewer, attr, False)
            except Exception:
                pass

    return viewer_ctx, viewer


def make_renderer(model, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
    if not HAVE_RENDERER:
        raise RuntimeError("This MuJoCo installation does not provide mujoco.Renderer.")

    cam_id = configure_pick_throw_camera(model)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    render_cam = mujoco.MjvCamera()
    render_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    render_cam.fixedcamid = cam_id
    return renderer, render_cam


def close_renderer_compat(renderer):
    """Close newer MuJoCo renderers; let older versions release on deletion."""
    if renderer is None:
        return

    close_fn = getattr(renderer, "close", None)
    if callable(close_fn):
        try:
            close_fn()
        except Exception as exc:
            # Cleanup must never abort the remaining randomized episodes.
            print(f"[warn] Renderer cleanup skipped: {exc}")


def render_corrected_frame(renderer, data, render_cam):
    renderer.update_scene(data, camera=render_cam)
    frame = renderer.render().astype(np.float32) / 255.0
    frame = np.clip(
        frame * VIDEO_BRIGHTNESS_GAIN + VIDEO_BRIGHTNESS_BIAS,
        0.0,
        1.0,
    )
    return (frame * 255).astype(np.uint8)


def append_video_frame(writer, frame):
    repeats = max(1, int(DUPLICATE_FRAMES))
    for _ in range(repeats):
        writer.append_data(frame)
    return repeats


def copy_data_for_recorded_tail(model, source):
    """Copy state so the extra recorded tail cannot affect grasp results."""
    target = mujoco.MjData(model)
    for name in (
        "qpos", "qvel", "act", "qacc_warmstart", "ctrl",
        "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat",
        "userdata", "eq_active",
    ):
        src = getattr(source, name, None)
        dst = getattr(target, name, None)
        if src is not None and dst is not None:
            try:
                dst[...] = src
            except (TypeError, ValueError):
                pass

    target.time = source.time
    mujoco.mj_forward(model, target)
    return target


# ---------------- Snapshot capture ----------------
def save_snapshot_png(model, data, out_png, width=960, height=540):
    if not HAVE_RENDERER:
        return False

    renderer = None
    try:
        renderer, render_cam = make_renderer(model, width=width, height=height)
        renderer.update_scene(data, camera=render_cam)
        imageio.imwrite(out_png, renderer.render())
        return True
    except Exception:
        return False
    finally:
        close_renderer_compat(renderer)


# ---------------- Plotting helpers ----------------
def _plot_timeseries(t, ys, labels, title, ylabel, out_png, ylim=None):
    t = np.asarray(t, float).reshape(-1)
    fig = plt.figure(figsize=(10.2, 3.9))
    ax = plt.gca()
    for y, lab in zip(ys, labels):
        y = np.asarray(y, float).reshape(-1)
        ax.plot(t, y, label=str(lab))
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)

def _plot_dual(t, y1, y2, lab1, lab2, title, y1lab, y2lab, out_png):
    t = np.asarray(t, float).reshape(-1)
    y1 = np.asarray(y1, float).reshape(-1)
    y2 = np.asarray(y2, float).reshape(-1)
    fig = plt.figure(figsize=(10.2, 3.9))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(t, y1, label=lab1)
    ax2.plot(t, y2, label=lab2)
    ax1.set_title(title)
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel(y1lab)
    ax2.set_ylabel(y2lab)
    ax1.grid(True, alpha=0.3)
    # combined legend
    lines = ax1.get_lines() + ax2.get_lines()
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)

def _plot_cost_components(t, cost_series, out_png, title="Cost components (running)"):
    t = np.asarray(t, float).reshape(-1)
    fig = plt.figure(figsize=(11.0, 4.2))
    ax = plt.gca()
    for k in ["Jd", "Jtheta", "Jcov", "Jfill", "Jr", "Jstall", "Jtotal"]:
        ax.plot(t, cost_series[k], label=k)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("accumulated / composed cost")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9, ncol=4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)

def _plot_final_cost_bars(cost_final_a, cost_final_b, label_a, label_b, out_png,
                          title="Final cost decomposition"):
    keys = ["Jd", "Jtheta", "Jcov", "Jfill", "Jr", "Jstall", "Jtotal"]
    a = [float(cost_final_a.get(k, 0.0)) for k in keys]
    b = [float(cost_final_b.get(k, 0.0)) for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig = plt.figure(figsize=(10.5, 4.0))
    ax = plt.gca()
    ax.bar(x - w/2, a, width=w, label=label_a)
    ax.bar(x + w/2, b, width=w, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20)
    ax.set_title(title)
    ax.set_ylabel("final cost component value")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close(fig)


# ---------------- Replay core ----------------
def replay_and_log(model, goal_xyz, ctrl_seq, cfg, episode_dir, tag="rollback",
                   show_view=True, record_to=None, debug_frame_to=None,
                   post_view_sec=POST_VIEW_SEC):
    """
    Returns a dict with:
      time series arrays, final metrics, final cost dict, and key timestamps.
    """
    d = mujoco.MjData(model)
    mujoco.mj_resetData(model, d)
    init_cylinder_pose(model, d, goal_xyz, cfg["freejoint_name"])
    mujoco.mj_forward(model, d)

    # IDs
    obj_bid = _mj_name2id_safe(model, mujoco.mjtObj.mjOBJ_BODY, cfg["obj_body"])
    if obj_bid < 0:
        raise RuntimeError(f"Object body '{cfg['obj_body']}' not found.")
    tip_sid = _mj_name2id_safe(model, mujoco.mjtObj.mjOBJ_SITE, cfg["tip_site"])
    if tip_sid < 0:
        raise RuntimeError(f"Tip site '{cfg['tip_site']}' not found. Use --tip-site to set it.")

    chain_bids = []
    for nm in cfg["chain_bodies"]:
        bid = _mj_name2id_safe(model, mujoco.mjtObj.mjOBJ_BODY, nm)
        if bid >= 0:
            chain_bids.append(bid)
    if len(chain_bids) == 0:
        print("[warn] No chain bodies found for enclosure metrics; Cov/Fill will be 0.")

    robot_geom_set, obj_geom_set = _robot_object_geom_sets(model, obj_bid)
    cam_id = configure_pick_throw_camera(model)

    # viewer (optional)
    viewer_ctx = None
    viewer = None
    if show_view and HAVE_MJV:
        try:
            viewer_ctx, viewer = launch_clean_passive_viewer(model, d, cam_id)
        except Exception as exc:
            print(f"[warn] Viewer unavailable: {exc}")
            viewer_ctx = None
            viewer = None

    # video recording (optional)
    renderer = None
    render_cam = None
    video_writer = None
    frames_written = 0
    video_stride = max(
        1,
        int((1.0 / VIDEO_FPS) / max(float(cfg["dt_sim"]), 1e-12)),
    )
    if record_to is not None:
        record_to = Path(record_to)
        record_to.parent.mkdir(parents=True, exist_ok=True)
        renderer, render_cam = make_renderer(model)

        test_frame = render_corrected_frame(renderer, d, render_cam)
        if debug_frame_to is None:
            debug_frame_to = record_to.with_name(f"debug_first_frame_{record_to.stem}.png")
        debug_frame_to = Path(debug_frame_to)
        debug_frame_to.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(debug_frame_to, test_frame)
        print(f"[VIDEO] first frame mean pixel = {test_frame.mean():.2f}")
        print(f"[VIDEO] debug frame saved: {debug_frame_to}")
        video_writer = imageio.get_writer(
            str(record_to),
            fps=VIDEO_FPS,
            quality=10,
            macro_block_size=1,
        )

    # logs
    T = int(ctrl_seq.shape[0])
    t_log = np.zeros(T, dtype=float)

    r_log = np.zeros(T, dtype=float)
    phi_log = np.zeros(T, dtype=float)
    phi_unwrap = np.zeros(T, dtype=float)
    dphi_log = np.zeros(T, dtype=float)
    theta_cum = np.zeros(T, dtype=float)

    cov_log = np.zeros(T, dtype=float)
    fill_log = np.zeros(T, dtype=float)

    I_contact = np.zeros(T, dtype=float)
    n_contact = np.zeros(T, dtype=float)
    fn_sum = np.zeros(T, dtype=float)

    I_near = np.zeros(T, dtype=float)
    jam_proxy = np.zeros(T, dtype=float)

    # cost accumulations
    cost = {"Jd":0.0, "Jtheta":0.0, "Jcov":0.0, "Jfill":0.0, "Jr":0.0, "Jstall":0.0, "Jtotal":0.0}
    cost_series = {k: np.zeros(T, dtype=float) for k in cost.keys()}

    # initial geometry
    p_obj = _body_pos(d, obj_bid)
    p_tip = _site_pos(d, tip_sid)
    phi0 = _wrap_angle_xz(p_tip, p_obj)

    phi_log[0] = phi0
    phi_unwrap[0] = phi0
    theta_cum[0] = 0.0

    # snapshots times (indices)
    snap_idx_start = 0
    snap_idx_final = T-1
    snap_idx_rollback = None
    most_negative_dphi = 0.0
    most_negative_i = None

    prev_u = None
    dt_sleep = float(cfg["dt_sim"]) * float(cfg["playback_slowdown"]) if cfg["playback_slowdown"] > 0 else 0.0

    try:
        for i in range(T):
            u = np.asarray(ctrl_seq[i], dtype=float).copy()
            d.ctrl[:] = u
            mujoco.mj_step(model, d)
            if viewer is not None:
                try:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = cam_id
                    viewer.sync()
                except Exception:
                    pass

            if video_writer is not None and (i % video_stride == 0):
                frame = render_corrected_frame(renderer, d, render_cam)
                frames_written += append_video_frame(video_writer, frame)

                if frames_written % 60 == 0:
                    print(f"[VIDEO] captured {frames_written} frames")

            if dt_sleep > 0 and cfg["show_view"]:
                time.sleep(dt_sleep)

            t_log[i] = float(d.time)

            # positions
            p_obj = _body_pos(d, obj_bid)
            p_tip = _site_pos(d, tip_sid)

            # distance & angles
            r = float(np.linalg.norm(p_tip - p_obj))
            r_log[i] = r

            phi = _wrap_angle_xz(p_tip, p_obj)
            phi_log[i] = phi

            if i > 0:
                dphi = _unwrap_increment(phi_log[i-1], phi)
                dphi_log[i] = dphi
                phi_unwrap[i] = phi_unwrap[i-1] + dphi
                # directional wrap progress (positive only)
                theta_cum[i] = theta_cum[i-1] + max(0.0, float(dphi))
            else:
                dphi_log[i] = 0.0
                phi_unwrap[i] = phi0
                theta_cum[i] = 0.0

            # detect rollback moment (most negative dphi)
            if i > 0 and float(dphi_log[i]) < float(most_negative_dphi):
                most_negative_dphi = float(dphi_log[i])
                most_negative_i = int(i)

            # enclosure metrics
            if len(chain_bids) > 0:
                pts_xz = []
                for bid in chain_bids:
                    pb = _body_pos(d, bid)
                    pts_xz.append([float(pb[0]), float(pb[2])])
                cov, fill, counts = compute_cov_fill_from_chain_points(
                    pts_xz, obj_xz=[float(p_obj[0]), float(p_obj[2])],
                    n_bins=int(cfg["cov_bins"]),
                    r_min=float(cfg["cov_rmin"]),
                    r_max=float(cfg["cov_rmax"])
                )
            else:
                cov, fill, counts = 0.0, 0.0, np.zeros(int(cfg["cov_bins"]), dtype=int)

            cov_log[i] = float(cov)
            fill_log[i] = float(fill)

            # contact metrics
            in_c, n_c, fn = _contact_metrics(model, d, robot_geom_set, obj_geom_set)
            I_contact[i] = 1.0 if in_c else 0.0
            n_contact[i] = float(n_c)
            fn_sum[i] = float(fn)

            # near/jam proxies
            near = (r <= float(cfg["near_r"]))
            I_near[i] = 1.0 if near else 0.0
            jam_proxy[i] = 1.0 if (near and abs(float(dphi_log[i])) < float(cfg["stall_dphi_eps"])) else 0.0

            # cost update
            du = None if prev_u is None else (u - prev_u)
            prev_u = u.copy()

            update_cost_accumulators(
                cost, r=r, theta_cum=float(theta_cum[i]), cov=float(cov), fill=float(fill),
                u=u, du=du, in_contact=bool(in_c), near=bool(near), dphi=float(dphi_log[i]),
                w_dist=float(cfg["w_dist"]), r_target=float(cfg["r_target"]),
                w_theta=float(cfg["w_theta"]), w_cov=float(cfg["w_cov"]), w_fill=float(cfg["w_fill"]),
                w_u=float(cfg["w_u"]), w_du=float(cfg["w_du"]),
                w_stall=float(cfg["w_stall"]), stall_dphi_eps=float(cfg["stall_dphi_eps"])
            )

            for k in cost_series.keys():
                cost_series[k][i] = float(cost[k])

        # Match the pick-and-throw recorded tail without changing any grasp
        # metric: the additional zero-control steps use a copied state.
        if video_writer is not None:
            extra_steps = int(max(float(post_view_sec), 0.0) / float(cfg["dt_sim"]))
            if extra_steps > 0:
                print(
                    f"[TAIL] recording extra {post_view_sec:.2f} sec "
                    f"= {extra_steps} sim steps"
                )
                tail_data = copy_data_for_recorded_tail(model, d)
                tail_data.ctrl[:] = 0.0

                for j in range(extra_steps):
                    mujoco.mj_step(model, tail_data)
                    mujoco.mj_forward(model, tail_data)
                    if j % video_stride == 0:
                        frame = render_corrected_frame(renderer, tail_data, render_cam)
                        frames_written += append_video_frame(video_writer, frame)

    finally:
        if viewer_ctx is not None:
            try:
                viewer_ctx.__exit__(None, None, None)
            except Exception:
                pass
        try:
            if video_writer is not None:
                print(f"[VIDEO] total frames = {frames_written}")
                video_writer.close()
        finally:
            close_renderer_compat(renderer)
            renderer = None

        if video_writer is not None:
            if frames_written > 0:
                print(f"[VIDEO] Saved MuJoCo-rendered video: {record_to}")
            else:
                print("[VIDEO] No frames captured. Video not saved.")

    # finalize rollback snapshot index
    if most_negative_i is not None:
        snap_idx_rollback = most_negative_i
    else:
        snap_idx_rollback = int(T * 0.6)

    # final polar occupancy for last step (recompute counts)
    if len(chain_bids) > 0:
        pts_xz = []
        for bid in chain_bids:
            pb = _body_pos(d, bid)
            pts_xz.append([float(pb[0]), float(pb[2])])
        _, _, counts_final = compute_cov_fill_from_chain_points(
            pts_xz, obj_xz=[float(p_obj[0]), float(p_obj[2])],
            n_bins=int(cfg["cov_bins"]),
            r_min=float(cfg["cov_rmin"]),
            r_max=float(cfg["cov_rmax"])
        )
    else:
        counts_final = np.zeros(int(cfg["cov_bins"]), dtype=int)

    # snapshots: start / rollback / final (re-sim quickly to those indices for clean rendering)
    snaps_dir = _ensure_dir(os.path.join(episode_dir, "snapshots"))
    if HAVE_RENDERER:
        # quick deterministic re-run without sleeping/viewer to capture frames at indices
        dd = mujoco.MjData(model)
        mujoco.mj_resetData(model, dd)
        init_cylinder_pose(model, dd, goal_xyz, cfg["freejoint_name"])
        mujoco.mj_forward(model, dd)

        snap_set = {snap_idx_start: "start", snap_idx_rollback: "rollback", snap_idx_final: "final"}
        for i in range(T):
            dd.ctrl[:] = ctrl_seq[i]
            mujoco.mj_step(model, dd)
            if i in snap_set:
                name = snap_set[i]
                out_png = os.path.join(snaps_dir, f"{tag}_{name}.png")
                ok = save_snapshot_png(model, dd, out_png, width=int(cfg["snap_w"]), height=int(cfg["snap_h"]))
                if ok:
                    pass

    # save CSV time series
    csv_path = os.path.join(episode_dir, f"{tag}_timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s",
            "r_tip_obj",
            "phi",
            "phi_unwrap",
            "dphi",
            "theta_cum",
            "cov",
            "fill",
            "I_contact",
            "n_contact",
            "Fn_sum",
            "I_near",
            "jam_proxy",
            "Jd", "Jtheta", "Jcov", "Jfill", "Jr", "Jstall", "Jtotal"
        ])
        for i in range(T):
            w.writerow([
                float(t_log[i]),
                float(r_log[i]),
                float(phi_log[i]),
                float(phi_unwrap[i]),
                float(dphi_log[i]),
                float(theta_cum[i]),
                float(cov_log[i]),
                float(fill_log[i]),
                float(I_contact[i]),
                float(n_contact[i]),
                float(fn_sum[i]),
                float(I_near[i]),
                float(jam_proxy[i]),
                float(cost_series["Jd"][i]),
                float(cost_series["Jtheta"][i]),
                float(cost_series["Jcov"][i]),
                float(cost_series["Jfill"][i]),
                float(cost_series["Jr"][i]),
                float(cost_series["Jstall"][i]),
                float(cost_series["Jtotal"][i]),
            ])

    return {
        "t": t_log,
        "r": r_log,
        "phi": phi_log,
        "phi_unwrap": phi_unwrap,
        "dphi": dphi_log,
        "theta_cum": theta_cum,
        "cov": cov_log,
        "fill": fill_log,
        "I_contact": I_contact,
        "n_contact": n_contact,
        "Fn_sum": fn_sum,
        "I_near": I_near,
        "jam_proxy": jam_proxy,
        "cost_series": cost_series,
        "cost_final": {k: float(cost_series[k][-1]) for k in cost_series.keys()},
        "counts_final": counts_final,
        "rollback_index": int(snap_idx_rollback),
        "rollback_time": float(t_log[int(snap_idx_rollback)]),
        "tag": tag,
        "csv": csv_path
    }


def ablate_no_rollback_from_logged_phi(ctrl_seq, dphi_log):
    """
    Replay-time ablation:
      - if dphi_log[i] < 0 (rollback happening), freeze ctrl (hold previous) to suppress it.
    This is intentionally conservative and doesn't require knowing actuator semantics.
    """
    ctrl_seq = np.asarray(ctrl_seq, float).copy()
    dphi_log = np.asarray(dphi_log, float).reshape(-1)
    out = ctrl_seq.copy()
    for i in range(1, out.shape[0]):
        if float(dphi_log[i]) < 0.0:
            out[i] = out[i-1]
    return out


# ---------------- Main driver ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Replay saved controls and export paper-ready plot packs (no CMA rerun)."
    )
    parser.add_argument("--xml", type=str, default=XML_PATH)
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR)
    parser.add_argument("--obj-body", type=str, default=OBJ_BODY_NAME)
    parser.add_argument("--freejoint", type=str, default=CYL_FREEJOINT_NAME)
    parser.add_argument("--tip-site", type=str, default=DEFAULT_TIP_SITE)
    parser.add_argument("--chain", type=str, nargs="*", default=DEFAULT_CHAIN,
                        help="Body names used for enclosure metrics (default B12..B23).")

    parser.add_argument("--no-view", action="store_true", help="Disable viewer (faster).")
    parser.add_argument("--record-video", action="store_true",
                        help="Record pick-and-throw-style 1920x1080 MP4 videos.")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Video/debug-frame directory; defaults to --save-dir.")
    parser.add_argument("--post-view-sec", type=float, default=POST_VIEW_SEC,
                        help="Extra zero-control seconds appended only to each MP4.")
    parser.add_argument("--ablate-no-rollback", action="store_true",
                        help="Also export a replay-time no-rollback ablation variant (no rerun).")

    # enclosure settings
    parser.add_argument("--cov-bins", type=int, default=36)
    parser.add_argument("--cov-rmin", type=float, default=0.0)
    parser.add_argument("--cov-rmax", type=float, default=0.08)

    # jam/near thresholds
    parser.add_argument("--near-r", type=float, default=0.03)
    parser.add_argument("--stall-dphi-eps", type=float, default=0.002)

    # cost weights (tune later)
    parser.add_argument("--w-dist", type=float, default=1.0)
    parser.add_argument("--r-target", type=float, default=0.0)
    parser.add_argument("--w-theta", type=float, default=1.0)
    parser.add_argument("--w-cov", type=float, default=1.0)
    parser.add_argument("--w-fill", type=float, default=1.0)
    parser.add_argument("--w-u", type=float, default=0.05)
    parser.add_argument("--w-du", type=float, default=0.01)
    parser.add_argument("--w-stall", type=float, default=0.5)

    # snapshots
    parser.add_argument("--snap-w", type=int, default=960)
    parser.add_argument("--snap-h", type=int, default=540)

    args = parser.parse_args()

    summary, summary_path = _load_summary(args.save_dir)

    DT_SIM            = float(summary.get("DT_SIM", 0.002))
    PLAYBACK_SLOWDOWN = float(summary.get("PLAYBACK_SLOWDOWN", 0.0))  # default off for speed
    episodes = summary.get("episodes_AB", None)

    # handle single-episode summaries gracefully
    if episodes is None:
        # build pseudo list of one episode if possible
        goal_xyz = summary.get("goal_xyz", summary.get("goal_xyz_fixed", None))
        if goal_xyz is None:
            raise RuntimeError("Summary has no episodes_AB and no goal_xyz; can't replay.")
        episodes = [{
            "abs_x": abs(float(goal_xyz[0])),
            "z": float(goal_xyz[2]),
            "RIGHT": {"ctrl_file": summary.get("ctrl_file", "episode_RIGHT_ctrl.npy")},
        }]

    modelR = mujoco.MjModel.from_xml_path(args.xml)
    modelL = mujoco.MjModel.from_xml_path(args.xml)
    modelR.opt.timestep = DT_SIM
    modelL.opt.timestep = DT_SIM

    cfg = {
        "dt_sim": DT_SIM,
        "playback_slowdown": PLAYBACK_SLOWDOWN,
        "obj_body": args.obj_body,
        "freejoint_name": args.freejoint,
        "tip_site": args.tip_site,
        "chain_bodies": args.chain,
        "cov_bins": int(args.cov_bins),
        "cov_rmin": float(args.cov_rmin),
        "cov_rmax": float(args.cov_rmax),
        "near_r": float(args.near_r),
        "stall_dphi_eps": float(args.stall_dphi_eps),
        "w_dist": float(args.w_dist),
        "r_target": float(args.r_target),
        "w_theta": float(args.w_theta),
        "w_cov": float(args.w_cov),
        "w_fill": float(args.w_fill),
        "w_u": float(args.w_u),
        "w_du": float(args.w_du),
        "w_stall": float(args.w_stall),
        "snap_w": int(args.snap_w),
        "snap_h": int(args.snap_h),
        "show_view": (not args.no_view),
    }

    pack_root = _ensure_dir(os.path.join(args.save_dir, "plot_pack"))
    record_root = Path(args.record_dir) if args.record_dir is not None else Path(args.save_dir)
    if args.record_video:
        record_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] Plot pack root: {pack_root}")
    print(f"[info] Summary: {summary_path}")
    print(f"[info] Episodes: {len(episodes)}")
    print(f"[info] Running all {len(episodes)} randomized coordinate episodes in this invocation.")

    aggregate_rows = []  # per episode/side/tag summary

    # Keep the original full randomized run: no episode slicing or early exit.
    # Every coordinate in episodes_AB is replayed for its saved RIGHT/LEFT files.
    for idx, ep in enumerate(episodes, start=1):
        ax_abs = float(ep.get("abs_x", ep.get("x", 0.0)))
        gz = float(ep.get("z", 0.75))

        goalR = np.array([+ax_abs, 0.0, gz], float)
        goalL = np.array([-ax_abs, 0.0, gz], float)

        right_ctrl_file = ep.get("RIGHT", {}).get("ctrl_file", f"episode_ep{idx}_RIGHT_ctrl.npy")
        left_ctrl_file  = ep.get("LEFT",  {}).get("ctrl_file",  f"episode_ep{idx}_LEFT_ctrl.npy")

        # ---------- RIGHT ----------
        pathR = os.path.join(args.save_dir, right_ctrl_file)
        if os.path.isfile(pathR):
            ctrlR = np.load(pathR)
            ep_dir = _ensure_dir(os.path.join(pack_root, f"episode_ep{idx}_RIGHT"))
            print(f"\n== PlotPack Episode {idx} RIGHT | goal=({+ax_abs:.4f}, {gz:.4f}) | steps={ctrlR.shape[0]} ==")

            # Replay rollback policy
            video_path = (
                record_root / f"replay_ep{idx}_RIGHT_mujoco_render.mp4"
                if args.record_video else None
            )
            debug_path = (
                record_root / f"debug_first_frame_ep{idx}_RIGHT.png"
                if args.record_video else None
            )
            outA = replay_and_log(modelR, goalR, ctrlR, cfg, ep_dir, tag="rollback",
                                  show_view=(not args.no_view),
                                  record_to=video_path,
                                  debug_frame_to=debug_path,
                                  post_view_sec=args.post_view_sec)

            # Export plots for rollback
            _ensure_dir(os.path.join(ep_dir, "plots"))
            plots_dir = os.path.join(ep_dir, "plots")

            _plot_timeseries(outA["t"], [outA["r"]], ["rollback"], "r(t): tip-object distance", "r [m]",
                             os.path.join(plots_dir, "r_t.png"))
            _plot_timeseries(outA["t"], [outA["phi"], outA["phi_unwrap"]], ["phi", "phi_unwrap"],
                             "phi(t) and unwrapped phi", "angle [rad]",
                             os.path.join(plots_dir, "phi_unwrap.png"))
            _plot_timeseries(outA["t"], [outA["theta_cum"]], ["Theta_cum"], "Theta_cum(t): cumulative wrap", "rad (cum)",
                             os.path.join(plots_dir, "theta_cum.png"))
            _plot_timeseries(outA["t"], [outA["cov"], outA["fill"]], ["Cov", "Fill"],
                             "Enclosure proxies: Coverage and Fill", "score [0..1]",
                             os.path.join(plots_dir, "cov_fill.png"), ylim=(-0.02, 1.02))
            _plot_timeseries(outA["t"], [outA["I_contact"], outA["n_contact"]], ["I_contact", "n_contact"],
                             "Contact regime", "contact / count",
                             os.path.join(plots_dir, "contacts.png"))
            _plot_timeseries(outA["t"], [outA["Fn_sum"]], ["Sum Fn"], "Sum normal force (robot-object)", "force (arb)",
                             os.path.join(plots_dir, "force_sum.png"))
            _plot_dual(outA["t"], outA["I_near"], np.abs(outA["dphi"]),
                       "I_near", "|dphi|",
                       "Jam/slip proxy: near vs angular progress",
                       "I_near (0/1)", "|dphi| [rad/step]",
                       os.path.join(plots_dir, "near_vs_dphi.png"))
            _plot_cost_components(outA["t"], outA["cost_series"],
                                  os.path.join(plots_dir, "cost_components.png"))

            polar_occupancy_plot(outA["counts_final"],
                                 os.path.join(plots_dir, "polar_occupancy_final.png"),
                                 title="Polar occupancy (final) - rollback")

            # Optional ablation: no-rollback variant
            outB = None
            if args.ablate_no_rollback:
                ctrlNR = ablate_no_rollback_from_logged_phi(ctrlR, outA["dphi"])
                outB = replay_and_log(modelR, goalR, ctrlNR, cfg, ep_dir, tag="no_rollback",
                                      show_view=False)

                # Compare r(t) and Theta_cum
                _plot_timeseries(outA["t"], [outA["r"], outB["r"]], ["rollback", "no_rollback"],
                                 "r(t) comparison", "r [m]",
                                 os.path.join(plots_dir, "r_compare.png"))
                _plot_timeseries(outA["t"], [outA["theta_cum"], outB["theta_cum"]], ["rollback", "no_rollback"],
                                 "Theta_cum(t) comparison", "rad (cum)",
                                 os.path.join(plots_dir, "theta_cum_compare.png"))
                _plot_timeseries(outA["t"], [outA["cov"], outB["cov"]], ["rollback", "no_rollback"],
                                 "Coverage comparison", "score [0..1]",
                                 os.path.join(plots_dir, "cov_compare.png"), ylim=(-0.02, 1.02))
                _plot_timeseries(outA["t"], [outA["fill"], outB["fill"]], ["rollback", "no_rollback"],
                                 "Fill comparison", "score [0..1]",
                                 os.path.join(plots_dir, "fill_compare.png"), ylim=(-0.02, 1.02))

                _plot_final_cost_bars(outA["cost_final"], outB["cost_final"],
                                      "rollback", "no_rollback",
                                      os.path.join(plots_dir, "final_cost_decomp.png"))

                polar_occupancy_plot(outB["counts_final"],
                                     os.path.join(plots_dir, "polar_occupancy_final_no_rollback.png"),
                                     title="Polar occupancy (final) - no_rollback")

            # write summary JSON for episode
            ep_summary = {
                "episode_index": idx,
                "side": "RIGHT",
                "goal_xyz": [float(goalR[0]), float(goalR[1]), float(goalR[2])],
                "rollback": {
                    "rollback_time": float(outA["rollback_time"]),
                    "final": outA["cost_final"],
                    "final_theta_cum": float(outA["theta_cum"][-1]),
                    "final_cov": float(outA["cov"][-1]),
                    "final_fill": float(outA["fill"][-1]),
                    "final_r": float(outA["r"][-1]),
                    "final_contact_frac": float(np.mean(outA["I_contact"])),
                    "csv": outA["csv"],
                },
                "no_rollback": None if outB is None else {
                    "final": outB["cost_final"],
                    "final_theta_cum": float(outB["theta_cum"][-1]),
                    "final_cov": float(outB["cov"][-1]),
                    "final_fill": float(outB["fill"][-1]),
                    "final_r": float(outB["r"][-1]),
                    "final_contact_frac": float(np.mean(outB["I_contact"])),
                    "csv": outB["csv"],
                    "note": "Replay-time ablation: ctrl held constant when dphi<0 (no rerun)."
                }
            }
            with open(os.path.join(ep_dir, "plot_pack_summary.json"), "w") as f:
                json.dump(ep_summary, f, indent=2)

            # aggregate table rows
            aggregate_rows.append({
                "episode": idx, "side": "RIGHT", "tag": "rollback",
                "final_theta_cum": float(outA["theta_cum"][-1]),
                "final_cov": float(outA["cov"][-1]),
                "final_fill": float(outA["fill"][-1]),
                "final_r": float(outA["r"][-1]),
                "contact_frac": float(np.mean(outA["I_contact"])),
                "Fn_mean": float(np.mean(outA["Fn_sum"])),
                "Jtotal": float(outA["cost_final"]["Jtotal"]),
            })
            if outB is not None:
                aggregate_rows.append({
                    "episode": idx, "side": "RIGHT", "tag": "no_rollback",
                    "final_theta_cum": float(outB["theta_cum"][-1]),
                    "final_cov": float(outB["cov"][-1]),
                    "final_fill": float(outB["fill"][-1]),
                    "final_r": float(outB["r"][-1]),
                    "contact_frac": float(np.mean(outB["I_contact"])),
                    "Fn_mean": float(np.mean(outB["Fn_sum"])),
                    "Jtotal": float(outB["cost_final"]["Jtotal"]),
                })

        else:
            print(f"[skip] Missing RIGHT ctrl for ep {idx}: {pathR}")

        # ---------- LEFT ----------
        pathL = os.path.join(args.save_dir, left_ctrl_file)
        if os.path.isfile(pathL):
            ctrlL = np.load(pathL)
            ep_dir = _ensure_dir(os.path.join(pack_root, f"episode_ep{idx}_LEFT"))
            print(f"== PlotPack Episode {idx} LEFT  | goal=({-ax_abs:.4f}, {gz:.4f}) | steps={ctrlL.shape[0]} ==")

            video_path = (
                record_root / f"replay_ep{idx}_LEFT_mujoco_render.mp4"
                if args.record_video else None
            )
            debug_path = (
                record_root / f"debug_first_frame_ep{idx}_LEFT.png"
                if args.record_video else None
            )
            outA = replay_and_log(modelL, goalL, ctrlL, cfg, ep_dir, tag="rollback",
                                  show_view=(not args.no_view),
                                  record_to=video_path,
                                  debug_frame_to=debug_path,
                                  post_view_sec=args.post_view_sec)

            _ensure_dir(os.path.join(ep_dir, "plots"))
            plots_dir = os.path.join(ep_dir, "plots")

            _plot_timeseries(outA["t"], [outA["r"]], ["rollback"], "r(t): tip-object distance", "r [m]",
                             os.path.join(plots_dir, "r_t.png"))
            _plot_timeseries(outA["t"], [outA["phi"], outA["phi_unwrap"]], ["phi", "phi_unwrap"],
                             "phi(t) and unwrapped phi", "angle [rad]",
                             os.path.join(plots_dir, "phi_unwrap.png"))
            _plot_timeseries(outA["t"], [outA["theta_cum"]], ["Theta_cum"], "Theta_cum(t): cumulative wrap", "rad (cum)",
                             os.path.join(plots_dir, "theta_cum.png"))
            _plot_timeseries(outA["t"], [outA["cov"], outA["fill"]], ["Cov", "Fill"],
                             "Enclosure proxies: Coverage and Fill", "score [0..1]",
                             os.path.join(plots_dir, "cov_fill.png"), ylim=(-0.02, 1.02))
            _plot_timeseries(outA["t"], [outA["I_contact"], outA["n_contact"]], ["I_contact", "n_contact"],
                             "Contact regime", "contact / count",
                             os.path.join(plots_dir, "contacts.png"))
            _plot_timeseries(outA["t"], [outA["Fn_sum"]], ["Sum Fn"], "Sum normal force (robot-object)", "force (arb)",
                             os.path.join(plots_dir, "force_sum.png"))
            _plot_dual(outA["t"], outA["I_near"], np.abs(outA["dphi"]),
                       "I_near", "|dphi|",
                       "Jam/slip proxy: near vs angular progress",
                       "I_near (0/1)", "|dphi| [rad/step]",
                       os.path.join(plots_dir, "near_vs_dphi.png"))
            _plot_cost_components(outA["t"], outA["cost_series"],
                                  os.path.join(plots_dir, "cost_components.png"))

            polar_occupancy_plot(outA["counts_final"],
                                 os.path.join(plots_dir, "polar_occupancy_final.png"),
                                 title="Polar occupancy (final) - rollback")

            outB = None
            if args.ablate_no_rollback:
                ctrlNR = ablate_no_rollback_from_logged_phi(ctrlL, outA["dphi"])
                outB = replay_and_log(modelL, goalL, ctrlNR, cfg, ep_dir, tag="no_rollback",
                                      show_view=False)

                _plot_timeseries(outA["t"], [outA["r"], outB["r"]], ["rollback", "no_rollback"],
                                 "r(t) comparison", "r [m]",
                                 os.path.join(plots_dir, "r_compare.png"))
                _plot_timeseries(outA["t"], [outA["theta_cum"], outB["theta_cum"]], ["rollback", "no_rollback"],
                                 "Theta_cum(t) comparison", "rad (cum)",
                                 os.path.join(plots_dir, "theta_cum_compare.png"))
                _plot_final_cost_bars(outA["cost_final"], outB["cost_final"],
                                      "rollback", "no_rollback",
                                      os.path.join(plots_dir, "final_cost_decomp.png"))

            ep_summary = {
                "episode_index": idx,
                "side": "LEFT",
                "goal_xyz": [float(goalL[0]), float(goalL[1]), float(goalL[2])],
                "rollback": {
                    "rollback_time": float(outA["rollback_time"]),
                    "final": outA["cost_final"],
                    "final_theta_cum": float(outA["theta_cum"][-1]),
                    "final_cov": float(outA["cov"][-1]),
                    "final_fill": float(outA["fill"][-1]),
                    "final_r": float(outA["r"][-1]),
                    "final_contact_frac": float(np.mean(outA["I_contact"])),
                    "csv": outA["csv"],
                },
                "no_rollback": None if outB is None else {
                    "final": outB["cost_final"],
                    "final_theta_cum": float(outB["theta_cum"][-1]),
                    "final_cov": float(outB["cov"][-1]),
                    "final_fill": float(outB["fill"][-1]),
                    "final_r": float(outB["r"][-1]),
                    "final_contact_frac": float(np.mean(outB["I_contact"])),
                    "csv": outB["csv"],
                    "note": "Replay-time ablation: ctrl held constant when dphi<0 (no rerun)."
                }
            }
            with open(os.path.join(ep_dir, "plot_pack_summary.json"), "w") as f:
                json.dump(ep_summary, f, indent=2)

            aggregate_rows.append({
                "episode": idx, "side": "LEFT", "tag": "rollback",
                "final_theta_cum": float(outA["theta_cum"][-1]),
                "final_cov": float(outA["cov"][-1]),
                "final_fill": float(outA["fill"][-1]),
                "final_r": float(outA["r"][-1]),
                "contact_frac": float(np.mean(outA["I_contact"])),
                "Fn_mean": float(np.mean(outA["Fn_sum"])),
                "Jtotal": float(outA["cost_final"]["Jtotal"]),
            })
            if outB is not None:
                aggregate_rows.append({
                    "episode": idx, "side": "LEFT", "tag": "no_rollback",
                    "final_theta_cum": float(outB["theta_cum"][-1]),
                    "final_cov": float(outB["cov"][-1]),
                    "final_fill": float(outB["fill"][-1]),
                    "final_r": float(outB["r"][-1]),
                    "contact_frac": float(np.mean(outB["I_contact"])),
                    "Fn_mean": float(np.mean(outB["Fn_sum"])),
                    "Jtotal": float(outB["cost_final"]["Jtotal"]),
                })

        else:
            print(f"[skip] Missing LEFT ctrl for ep {idx}: {pathL}")

    # -------- aggregate CSV + aggregate plots (distributions) --------
    agg_csv = os.path.join(pack_root, "aggregate_metrics.csv")
    with open(agg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "side", "tag", "final_theta_cum", "final_cov", "final_fill", "final_r",
                    "contact_frac", "Fn_mean", "Jtotal"])
        for r in aggregate_rows:
            w.writerow([r["episode"], r["side"], r["tag"],
                        r["final_theta_cum"], r["final_cov"], r["final_fill"], r["final_r"],
                        r["contact_frac"], r["Fn_mean"], r["Jtotal"]])
    print(f"\n[info] Wrote aggregate metrics: {agg_csv}")

    # simple distribution plots (if both tags exist)
    tags = sorted(list(set([r["tag"] for r in aggregate_rows])))
    if len(tags) >= 1:
        def collect(metric, tag):
            return [r[metric] for r in aggregate_rows if r["tag"] == tag]

        for metric, ylabel, title in [
            ("final_theta_cum", "Theta_cum(final)", "Distribution of Theta_cum(final)"),
            ("final_cov", "Cov(final)", "Distribution of Coverage(final)"),
            ("final_fill", "Fill(final)", "Distribution of Fill(final)"),
            ("final_r", "r(final) [m]", "Distribution of r(final)"),
            ("Jtotal", "Jtotal(final)", "Distribution of final total cost"),
        ]:
            fig = plt.figure(figsize=(9.2, 3.9))
            ax = plt.gca()
            data = [collect(metric, t) for t in tags]
            ax.boxplot(data, labels=tags, showmeans=True)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(pack_root, f"dist_{metric}.png"), dpi=250)
            plt.close(fig)

    print("[done] Plot packs exported.")


if __name__ == "__main__":
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
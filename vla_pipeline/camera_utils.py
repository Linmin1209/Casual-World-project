"""Attach tool cameras for RGB capture without switching observation_mode to pixel."""
from __future__ import annotations

from causal_world.envs.robot.camera import Camera

# Same poses as CausalWorld.__init__ (pixel mode).
_TOOL_CAMERA_POSES = [
    ([0.2496, 0.2458, 0.58], [0.3760, 0.8690, -0.2918, -0.1354]),
    ([0.0047, -0.2834, 0.58], [0.9655, -0.0098, -0.0065, -0.2603]),
    ([-0.2470, 0.2513, 0.50], [-0.3633, 0.8686, -0.3141, 0.1220]),
]


def _render_client(env):
    client = env._pybullet_client_w_o_goal_id
    if client is None:
        client = env._pybullet_client_full_id
    if client is None:
        raise RuntimeError("No PyBullet client available for camera rendering.")
    return client


def ensure_tool_cameras(env) -> None:
    """
    Enable get_current_camera_observations() while keeping structured obs for PPO teacher.

    CausalWorld only constructs Camera objects when observation_mode=='pixel'.
    """
    if getattr(env, "_tool_cameras", None) is not None:
        return

    client = _render_client(env)
    cameras = [
        Camera(
            camera_position=pos,
            camera_orientation=orn,
            pybullet_client_id=client,
        )
        for pos, orn in _TOOL_CAMERA_POSES
    ]
    env._tool_cameras = cameras
    env._robot._tool_cameras = cameras
    # Structured mode never sets _cameras on TriFingerObservations.
    env._robot._robot_observations._cameras = cameras


def capture_tool_camera_rgb_uint8(env) -> list:
    """
    Raw uint8 HWC images from tool cameras.

    Do not use get_current_camera_observations() for saving/training RGB: in
    structured mode pixel stats are normalized to [-1, 1], which clips to black.
    """
    ensure_tool_cameras(env)
    robs = env._robot._robot_observations
    cameras = robs._cameras
    if not cameras:
        raise RuntimeError("Tool cameras not initialized.")
    indices = robs._camera_indicies
    return [cameras[int(i)].get_image() for i in indices]

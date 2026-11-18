"""Run the local pi05_libero policy and LIBERO in the same Python process."""
import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE_STEPS = {
    "libero_spatial": 400,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_STEPS, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0, help="Task index within the suite (default: 0).")
    parser.add_argument("--save", action="store_true", help="Save a two-camera MP4 of the episode.")
    return parser.parse_args()


def policy_observation(obs, instruction):
    import numpy as np
    from openpi_client import image_tools
    from robosuite.utils.transform_utils import quat2axisangle

    def camera(key):
        rgb = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))

    return {
        "observation/image": camera("agentview_image"),
        "observation/wrist_image": camera("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"].copy()),
            obs["robot0_gripper_qpos"],
        )),
        "prompt": instruction,
    }


def run_episode(env, policy, task, initial_state, max_steps, writer):
    import numpy as np
    import torch

    def save_frame(obs):
        if writer is None:
            return
        frame = np.concatenate((
            obs["agentview_image"][::-1, ::-1],
            obs["robot0_eye_in_hand_image"][::-1, ::-1],
        ), axis=1)
        writer.append_data(frame)

    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    save_frame(obs)

    action_queue = deque()
    for step in range(1, max_steps + 1):
        if not action_queue:
            with torch.inference_mode():
                actions = policy.infer(policy_observation(obs, task.language))["actions"]
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < 5 or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action chunk: {actions.shape}")
            action_queue.extend(actions[:5])
        obs, _, done, _ = env.step(action_queue.popleft().tolist())
        save_frame(obs)
        if env.check_success():
            return "success", step
        if done:
            return "failure", step
    return "failure", max_steps


def main():
    args = parse_args()
    os.environ.setdefault("OPENPI_DATA_HOME", str(ROOT / ".cache/openpi"))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero-config"))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "mesa")

    import numpy as np
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from pi05_policy import load_policy

    suite = benchmark.get_benchmark_dict()[args.suite]()
    if not 0 <= args.task_id < suite.n_tasks:
        raise SystemExit(f"--task-id must be between 0 and {suite.n_tasks - 1}")

    task = suite.get_task(args.task_id)
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    # These serialized initial states come from the bundled LIBERO submodule.
    initial_state = torch.load(init_path, weights_only=False, map_location="cpu")[0]

    torch.set_num_threads(2)
    seed = 7 + 1000 * args.task_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_description = "CUDA + CPU embeddings" if torch.cuda.is_available() else "CPU"
    print(f"Loading pi05_libero ({device_description})...", flush=True)
    policy = load_policy(ROOT / "pi05_libero", vision_on_cpu=False, embeddings_on_cpu=True)
    print("Policy ready.", flush=True)

    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=256, camera_widths=256)
    video_path = None
    try:
        env.seed(seed)
        print(f"{args.suite}, task {args.task_id}, trial 0: {task.language}", flush=True)
        video = nullcontext()
        if args.save:
            import imageio.v2 as imageio

            video_dir = ROOT / "outputs/libero_videos" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            video_dir.mkdir(parents=True)
            video_path = video_dir / f"task{args.task_id:02d}_trial000.mp4"
            video = imageio.get_writer(video_path, fps=10)
        with video as writer:
            status, steps = run_episode(env, policy, task, initial_state, SUITE_STEPS[args.suite], writer)
    finally:
        env.close()
    print(f"{status}, {steps} steps" + (f", video: {video_path}" if video_path else ""), flush=True)


if __name__ == "__main__":
    main()

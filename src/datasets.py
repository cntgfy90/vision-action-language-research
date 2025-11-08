import metaworld
import numpy as np


def collect_metaworld_trajectories(
    env_name="pick-place-v3", num_trajectories=100, max_steps=200
):
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name]()
    task = ml1.train_tasks[0]
    env.set_task(task)

    all_trajectories = []

    for _ in range(num_trajectories):
        obs, _ = env.reset()
        target_pos = obs[:3] + np.random.uniform(-0.1, 0.1, 3)

        trajectory_actions = []

        for step in range(max_steps):
            if step < max_steps * 0.7:
                action = target_pos - obs[:3]
                action = np.clip(action, -0.05, 0.05)
                action = np.append(action, [0.0])
            else:
                action = env.action_space.sample() * 0.5

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trajectory_actions.append(action)

            if done:
                break

        all_trajectories.append(np.array(trajectory_actions))

    env.close()
    return all_trajectories

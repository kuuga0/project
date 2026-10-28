"""List project Gym task IDs without starting Isaac Sim."""

import gymnasium as gym
import h2_plus_lab  # noqa: F401


if __name__ == "__main__":
    for task_id, spec in sorted(gym.registry.items()):
        entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(entry, str) and entry.startswith("h2_plus_lab."):
            print(f"{task_id}\n  {entry}")

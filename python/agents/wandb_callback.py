import wandb
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class WandbEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_data = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "success" in info:
            self.episode_data.append(info)

        # if "fail_due_to_distractor" in info:
        #     wandb.log({"episode/fail_due_to_distractor": int(info["fail_due_to_distractor"])}, step=self.num_timesteps)

        if self.n_calls % self.model.n_steps == 0 and self.episode_data:
            successes = [e["success"] for e in self.episode_data]
            r_errors = [e["r_error"] for e in self.episode_data]
            theta_errors = [e["theta_error"] for e in self.episode_data]
            steps = [e["steps_taken"] for e in self.episode_data]
            head_turns = [e["head_turns"] for e in self.episode_data]
            locomotion_steps = [e["locomotion_steps"] for e in self.episode_data]
            fail_due_to_distractors = [e["fail_due_to_distractor"] for e in self.episode_data]

            wandb.log({
                "episode/success_rate": np.mean(successes),
                "episode/avg_r_error": np.mean(r_errors),
                "episode/avg_theta_error": np.mean(theta_errors),
                "episode/avg_steps_taken": np.mean(steps),
                "episode/avg_head_turns": np.mean(head_turns),
                "episode/locomotion_steps": np.mean(locomotion_steps),
                "episode/fail_due_to_distractor": np.mean(fail_due_to_distractors)
            }, step=self.num_timesteps)

            self.episode_data.clear()

        return True
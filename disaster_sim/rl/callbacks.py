"""
Custom Callbacks for Stable Baselines 3.
Used for logging custom metrics (coverage, victims rescued) to TensorBoard.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # We want to log at the end of the episode
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            # Access info dict from the environment
            info = self.locals["infos"][0]
            
            coverage = info.get("coverage", 0.0)
            victims_remaining = info.get("victims_remaining", 0)
            battery = info.get("battery", 0.0)
            collisions = info.get("collisions", 0)
            
            self.logger.record("custom/coverage", coverage)
            self.logger.record("custom/victims_remaining", victims_remaining)
            self.logger.record("custom/battery_end", battery)
            self.logger.record("custom/collisions", collisions)
            
        return True

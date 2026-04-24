from typing import Dict

import numpy as np
from soccer_twos import AgentInterface

from imitation_player_utils import TorchPolicyActor, load_baseline_model


class BaselineTeacherAgent(AgentInterface):
    def __init__(self, env):
        super().__init__()
        self.actor = TorchPolicyActor(load_baseline_model())
        self.name = "BaselineTeacherTorch"

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        return self.actor.act(observation)

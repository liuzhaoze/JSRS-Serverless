from typing import Any, TypeVar, cast

import numpy as np
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class RandomTrainingStats(TrainingStats):
    pass


TRandomTrainingStats = TypeVar("TRandomTrainingStats", bound=RandomTrainingStats)


class RandomPolicy(BasePolicy[TRandomTrainingStats]):
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        logits = np.random.rand(len(batch.obs), self.action_space.n)
        result = Batch(act=logits.argmax(axis=-1))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRandomTrainingStats:
        """Since a random agent learns nothing, it returns an empty dict."""
        return RandomTrainingStats()

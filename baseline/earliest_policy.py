from typing import Any, TypeVar, cast

import numpy as np
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class EarliestTrainingStats(TrainingStats):
    pass


TEarliestTrainingStats = TypeVar("TEarliestTrainingStats", bound=EarliestTrainingStats)


class EarliestPolicy(BasePolicy[TEarliestTrainingStats]):
    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        waiting_times = batch.obs[:, -self.action_space.n :]
        result = Batch(act=waiting_times.argmin(axis=-1))
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TEarliestTrainingStats:
        """Since an earliest agent learns nothing, it returns an empty dict."""
        return EarliestTrainingStats()

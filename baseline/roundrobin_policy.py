from typing import Any, TypeVar, cast

import numpy as np
from tianshou.data import Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ActBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats


class RoundRobinTrainingStats(TrainingStats):
    pass


TRoundRobinTrainingStats = TypeVar("TRoundRobinTrainingStats", bound=RoundRobinTrainingStats)


class RoundRobinPolicy(BasePolicy[TRoundRobinTrainingStats]):
    def __init__(
        self,
        *,
        action_space,
        observation_space=None,
        action_scaling=False,
        action_bound_method="clip",
        lr_scheduler=None,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            lr_scheduler=lr_scheduler,
        )
        self.__counter = 0

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:
        result = Batch(act=np.full((len(batch.obs),), self.__counter))
        self.__counter = (self.__counter + 1) % self.action_space.n
        return cast(ActBatchProtocol, result)

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TRoundRobinTrainingStats:
        """Since a round robin agent learns nothing, it returns an empty dict."""
        return RoundRobinTrainingStats()

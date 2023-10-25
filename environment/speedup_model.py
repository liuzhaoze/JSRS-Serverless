from utils import load_hyperparameters


class SpeedupModel:
    hyperparameters = load_hyperparameters()
    A = hyperparameters["A"]
    sigma = hyperparameters["sigma"]

    @staticmethod
    def low_variance_model(n: int) -> float:
        if n > 2 * SpeedupModel.A - 1:
            return SpeedupModel.A
        elif n > SpeedupModel.A:
            return (
                SpeedupModel.A
                * n
                / (
                    SpeedupModel.sigma * (SpeedupModel.A - 0.5)
                    + n * (1 - 0.5 * SpeedupModel.sigma)
                )
            )
        elif n >= 1:
            return (
                SpeedupModel.A
                * n
                / (SpeedupModel.A + 0.5 * SpeedupModel.sigma * (n - 1))
            )
        else:
            raise ValueError("n must be greater than or equal to 1")

    @staticmethod
    def high_variance_model(n: int) -> float:
        if (
            n
            > SpeedupModel.A + SpeedupModel.A * SpeedupModel.sigma - SpeedupModel.sigma
        ):
            return SpeedupModel.A
        elif n >= 1:
            return (
                n
                * SpeedupModel.A
                * (SpeedupModel.sigma + 1)
                / (
                    SpeedupModel.A
                    + SpeedupModel.A * SpeedupModel.sigma
                    - SpeedupModel.sigma
                    + n * SpeedupModel.sigma
                )
            )
        else:
            raise ValueError("n must be greater than or equal to 1")

    @staticmethod
    def SU(n: int) -> float:
        if SpeedupModel.sigma > 1:
            return SpeedupModel.high_variance_model(n)
        elif SpeedupModel.sigma >= 0:
            return SpeedupModel.low_variance_model(n)
        else:
            raise ValueError("sigma must be greater than or equal to 0")

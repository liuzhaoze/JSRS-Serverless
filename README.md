# Get Started

ENGLISH / [简体中文](./README_CN.md)

## 1 Install Dependencies

> To be completed

## 2 Configuration

- Hyperparameters configuration file: `./config/hyperparameters.yml`
- Instance configuration file: `./config/instances.yml`

## 3 Training

```shell
python ./train.py
```

- Log files are saved in the `./runs` directory
- Model files are saved in the `./models` directory

## 4 Evaluation

```shell
python ./eval.py <model_path>
```

For example:

```shell
python ./eval.py ./models/Feb01_13-45-01.pth
```

## 5 TensorBoard

```shell
tensorboard --logdir=runs
```

## 6 Plot Results

```shell
python ./result/plot_result.py
```

## 7 Others

```shell
# Generate Spot instance prices
python ./generate_price.py

# View instance prices
python ./plot_price.py

# Various unit tests
python ./test.py
```

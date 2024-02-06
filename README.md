# 使用方法

## 1 安装依赖

> 待完善

## 2 配置

- 超参数配置文件：`./config/hyperparameters.yml`
- 实例配置文件：`./config/instances.yml`

## 3 训练

```shell
python ./train.py
```

- 日志文件保存在 `./runs` 目录下
- 模型文件保存在 `./models` 目录下

## 4 评估

```shell
python ./eval.py <model_path>
```

例如：

```shell
python ./eval.py ./models/Feb01_13-45-01.pth
```

## 5 TensorBoard

```shell
tensorboard --logdir=runs
```

## 6 结果绘制

```shell
python ./result/plot_result.py
```

## 7 其他

```shell
# 生成 Spot 实例价格
python ./generate_price.py

# 查看实例价格
python ./plot_price.py

# 各项单元测试
python ./test.py
```

#  MAML-Pytorch
PyTorch implementation of the supervised learning experiments from the paper:
[Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400).

> Version 1.0: Both `MiniImagenet` and `Omniglot` Datasets are supported! Have Fun~

> Version 2.0: Re-write meta learner and basic learner. Solved some serious bugs in version 1.0.

For Tensorflow Implementation, please visit official [HERE](https://github.com/cbfinn/maml) and simplier version [HERE](https://github.com/dragen1860/MAML-TensorFlow).

For First-Order Approximation Implementation, Reptile namely, please visit [HERE](https://github.com/dragen1860/Reptile-Pytorch).

![heart](res/heart.gif)

# Platform
- Python: 3.x
- PyTorch: 2.0+
- NVIDIA GPU（3090 单卡已验证，建议开启 TF32）

# 安装
- 使用 requirements.TXT 管理依赖
- 安装命令：

```bash
pip install -r requirements.TXT
```

# MiniImagenet


## Howto

For 5-way 1-shot exp., it allocates nearly 6GB GPU memory.

1. download `MiniImagenet` dataset from [here](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4), splitting: `train/val/test.csv` from [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet).
2. extract it like:
```shell
miniimagenet/
├── images
	├── n0210891500001298.jpg  
	├── n0287152500001298.jpg 
	...
├── test.csv
├── val.csv
└── train.csv


```
3. modify the `path` in `miniimagenet_train.py`:
```python
        mini = MiniImagenet('miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=10000, resize=args.imgsz)
		...
        mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=100, resize=args.imgsz)
```
to your actual data path.

4. just run `python miniimagenet_train.py` and the running screenshot is as follows:
![screenshot-miniimagetnet](res/mini-screen.png)

If your reproducation perf. is not so good, maybe you can enlarge your `training epoch` to get longer training. And MAML is notorious for its hard training. Therefore, this implementation only provide you a basic start point to begin your research.
and the performance below is true and achieved on my machine.

## Benchmark

| Model                               | Fine Tune | 5-way Acc. |        | 20-way Acc.|        |
|-------------------------------------|-----------|------------|--------|------------|--------|
|                                     |           | 1-shot     | 5-shot | 1-shot     | 5-shot |
| Matching Nets                       | N         | 43.56%     | 55.31% | 17.31%     | 22.69% |
| Meta-LSTM                           |           | 43.44%     | 60.60% | 16.70%     | 26.06% |
| MAML                                | Y         | 48.7%      | 63.11% | 16.49%     | 19.29% |
| **Ours**                            | Y         | 46.2%      | 60.3%	| -    		 | - 	|



# Omniglot

## Howto
- 运行：`python omniglot_train.py`
- 数据集：若本地已存在 `<root>/processed/images_evaluation` 与 `images_background` 则跳过下载；否则自动从官网下载并解压到 `<root>/processed`。
- 缓存：首次整理后会在根目录生成 `omniglot.npy` 以加速后续加载。
- 根目录：训练脚本默认根目录为 `omniglot`，可在 `OmniglotNShot(root=...)` 中自定义。
- 显存：5-way 1-shot 约占用 3GB；根据显存调小 `--task_num`。

## 训练示例
```bash
python omniglot_train.py \
  --epoch 40000 --n_way 5 --k_spt 1 --k_qry 15 --imgsz 28 \
  --task_num 32 --meta_lr 1e-3 --update_lr 0.4 --update_step 5
```

## 训练加速与日志（3090）
- 启用 cuDNN benchmark、TF32，并提升 matmul 精度。
- 使用 CPU pinned memory + 非阻塞 GPU 传输。
- 训练阶段每 50 步输出：training acc、avg_time/step(s)、throughput(img/s)。
- 测试阶段输出：Test acc、eval_time(s)。


# Refer to this Rep.
```
@misc{MAML_Pytorch,
  author = {Liangqu Long},
  title = {MAML-Pytorch Implementation},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}},
  commit = {master}
}
```

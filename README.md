# MAML-Pytorch

A PyTorch implementation of Model-Agnostic Meta-Learning (MAML) for learning purposes, adapted from [dragen1860/MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch).

## Overview

This project implements the supervised learning experiments from the MAML paper. It supports both Omniglot and MiniImagenet datasets with optimized training pipeline for modern PyTorch (2.0+).

### Key Features

- **Dual Dataset Support**: Omniglot and MiniImagenet
- **PyTorch 2.0+ Optimized**: TF32 support, cuDNN benchmark
- **GPU Acceleration**: Pinned memory, non-blocking transfers
- **Training Monitoring**: Real-time accuracy and throughput logging
- **Model Checkpointing**: Automatic best model saving
- **Mixed Precision Ready**: High precision matmul support

## Requirements

- Python 3.x
- PyTorch 2.0+
- NVIDIA GPU (RTX 3090 recommended with TF32 support)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Datasets

### Omniglot

The Omniglot dataset is automatically downloaded if not present at `<root>/processed/images_evaluation` and `images_background`. After first processing, `omniglot.npy` is cached for faster loading.

**Training:**
```bash
python omniglot_train.py \
  --epoch 40000 --n_way 5 --k_spt 1 --k_qry 15 --imgsz 28 \
  --task_num 32 --meta_lr 1e-3 --update_lr 0.4 --update_step 5
```

**Memory Usage**: ~3GB for 5-way 1-shot. Reduce `--task_num` if insufficient GPU memory.

**Arguments:**
- `--epoch`: Number of training epochs (default: 40000)
- `--n_way`: Number of classes per task (default: 5)
- `--k_spt`: Support set size (default: 1)
- `--k_query`: Query set size (default: 15)
- `--imgsz`: Image size (default: 28)
- `--task_num`: Meta-batch size (default: 32)
- `--meta_lr`: Meta-level learning rate (default: 1e-3)
- `--update_lr`: Task-level inner learning rate (default: 0.4)
- `--update_step`: Inner update steps (default: 5)
- `--update_step_test`: Finetuning steps (default: 10)

### MiniImagenet

Download MiniImagenet from [here](https://github.com/dragen1860/LearningToCompare-Pytorch/issues/4) and splits (train/val/test.csv) from [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet).

**Directory Structure:**
```
miniimagenet/
├── images/
│   ├── n0210891500001298.jpg
│   ├── n0287152500001298.jpg
│   └── ...
├── test.csv
├── val.csv
└── train.csv
```

**Training:**
```bash
python miniimagenet_train.py
```

**Note**: Modify the data path in `miniimagenet_train.py` to point to your actual MiniImagenet directory.

**Memory Usage**: ~6GB for 5-way 1-shot.

## Architecture

### Meta Learner (`meta.py`)

Core MAML implementation with:
- Task-level gradient updates
- Meta-level optimization
- Support/query set processing
- Finetuning capability

### Learner (`learner.py`)

Configurable neural network supporting:
- Convolutional layers
- Batch normalization
- Linear layers
- Various activation functions

### Data Loaders

- **OmniglotNShot** (`omniglotNShot.py`): N-shot learning data loader for Omniglot
- **MiniImagenet** (`MiniImagenet.py`): Data loader for MiniImagenet with episode generation

## Training Performance

### Omniglot

Training outputs every 100 steps:
- Step accuracy across all inner updates
- Average time per step
- Throughput (images/second)

Evaluation every 500 steps:
- Test accuracy
- Evaluation time
- Automatic checkpoint saving

### MiniImagenet

Training outputs every 30 steps:
- Training accuracy

Evaluation every 500 steps:
- Test accuracy with mean confidence intervals

## Performance Optimizations

### GPU Acceleration (RTX 3090)

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

### Memory Efficiency

- CPU pinned memory for faster GPU transfers
- Non-blocking data transfers
- Automatic checkpoint cleanup

## Checkpoint Management

- Saved to `checkpoints/` directory
- Named as `maml_step{step}_acc{accuracy}.pth`
- Best model preserved after training
- Includes: model state, accuracy, hyperparameters

## Citation

```bibtex
@misc{MAML_Pytorch,
  author = {Liangqu Long},
  title = {MAML-Pytorch Implementation},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/dragen1860/MAML-Pytorch}}
}
```

## References

- Original Paper: [Model-Agnostic Meta-Learning](https://arxiv.org/abs/1703.03400)
- Official TensorFlow Implementation: [cbfinn/maml](https://github.com/cbfinn/maml)
- First-Order Approximation (Reptile): [dragen1860/Reptile-Pytorch](https://github.com/dragen1860/Reptile-Pytorch)

## Notes

MAML is known for challenging training. Results may vary based on:
- GPU hardware
- Hyperparameter tuning
- Training duration

This implementation provides a starting point for MAML research and learning purposes.

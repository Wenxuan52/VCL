# DGM (PyTorch 迁移版)

这个目录把 `original_code/dgm` 的 TensorFlow1/Keras1 实现迁移到了 PyTorch 2.x。

## 1. 训练 continual VAE

```bash
cd /workspace/VCL
python original_code_pytorch/dgm/exp.py mnist noreg
python original_code_pytorch/dgm/exp.py mnist ewc 5.0
python original_code_pytorch/dgm/exp.py mnist laplace 5.0
python original_code_pytorch/dgm/exp.py mnist si 1.0
python original_code_pytorch/dgm/exp.py mnist onlinevi
```

常用参数：

- `--n-iter-override 20`：临时减少每个 task epoch 数，先做 smoke test。
- `--batch-size 50`，`--lr 1e-4`
- `--cpu`：强制 CPU。
- `--data-path /path/to/data`：notMNIST `.mat` 文件根目录（内部需要有 `notMNIST/notMNIST_small.mat`）。

### 产物位置

- 模型：`original_code_pytorch/dgm/save/<run_name>/checkpoint_*.pt`
- 训练生成图：`original_code_pytorch/dgm/figs/<run_name>/*.png`
- 结果：`original_code_pytorch/dgm/results/<run_name>.pkl`

## 2. 评估 test log-likelihood

```bash
python original_code_pytorch/dgm/eval_ll.py mnist noreg
python original_code_pytorch/dgm/eval_ll.py mnist ewc 5.0 --eval-k 500
```

- 输出可视化图：`original_code_pytorch/dgm/figs/visualisation/*.png`
- 输出结果：`original_code_pytorch/dgm/results/*_eval.pkl`

## 3. 分类器与 KL 指标可视化

先训练分类器：

```bash
python original_code_pytorch/dgm/classifier/train_classifier.py --data-name mnist --epochs 100
```

再跑 KL 指标：

```bash
python original_code_pytorch/dgm/eval_kl.py mnist noreg
```

输出：`original_code_pytorch/dgm/results/*_kl.pkl`

## 4. 推荐的快速验证流程（4090D）

```bash
python original_code_pytorch/dgm/exp.py mnist noreg --n-iter-override 3 --eval-k 20
python original_code_pytorch/dgm/eval_ll.py mnist noreg --eval-k 20
```

先验证端到端可跑通，再把 `--n-iter-override` 去掉跑完整实验。

# VCL `original_code/dgm/` 代码考古与架构逆向（可迁移版）

## 0) TL;DR（<=10行）
- 一句话：`dgm/` 通过“任务级别的多 head VAE + 共享生成器 trunk”在 MNIST 10 个单类任务上做持续学习；`onlinevi` 是最接近论文 VCL 的实现（对共享生成器参数做高斯变分后验，并在任务间更新先验）。
- 最关键 5 个文件/函数：`exp.py::main`、`alg/onlinevi.py::construct_optimizer`、`alg/onlinevi.py::KL_param`、`models/bayesian_generator.py::bayesian_mlp_layer`、`alg/eval_test_ll.py::IS_estimate`。
- 最关键 5 个张量/参数对象：`X_ph`、`batch_size_ph`、`gen_shared_l*_mu_* / log_sig_*`、`enc_{task}` 输出 `mu_qz/log_sig_qz`、`shared_prior_params`（numpy 字典）。
- 最关键 3 个循环：`for task in 1..10`（任务循环）→ `for iteration in 1..n_iter`（epoch）→ `for j in 0..N/batch`（mini-batch）。

## 1) 运行入口与复现步骤

### 1.1 真正入口脚本
- 训练入口：`original_code/dgm/exp.py`，命令 `python exp.py data_name method lbd`。支持 `method in [noreg, ewc, laplace, si, onlinevi]`。`onlinevi` 时 `lbd` 占位不用。  
- test-LL 评估入口：`original_code/dgm/eval_ll.py`，命令 `python eval_ll.py data_name method lbd`。  
- classifier uncertainty 入口：`original_code/dgm/eval_kl.py`，命令 `python eval_kl.py data_name method lbd`；先在 `classifier/train_classifier.py` 训练分类器权重。

### 1.2 默认超参与任务设定
- 训练全局默认值在 `exp.py` 顶部硬编码：`dimZ=50, dimH=500, batch_size=50, lr=1e-4, K_mc=10, checkpoint=-1`。
- 任务划分来自 `config.py`：MNIST 为 `labels=[[0],[1],...,[9]]`（10 个 task，每个 task 单类），`n_iter=200`，`dimX=784`，似然 `ll='bernoulli'`。
- 每 task 数据切分：对该 task 类别数据，训练集再切 90% train / 10% valid（`exp.py`）；测试集来自原测试 split。

### 1.3 环境与依赖假设
- README 明确：TensorFlow 1.0 + Keras 1.2.0，其他版本可能出 bug。
- Python 2 语法（`print` 语句、`xrange`、`cPickle`）和 TF1 图执行（placeholder/session）。
- 代码存在 `data_path = # TODO` 占位（`exp.py/eval_ll.py/eval_kl.py/classifier/train_classifier.py`），运行前必须手动填。

### 1.4 输出物与路径
- 模型参数：`save/<data>_<method>/checkpoint_<k>.pkl`（逐任务保存一次，`utils.save_params` 保存所有 `tf.trainable_variables` 到字典）。
- 生成图：训练过程每 task 输出 `figs/<path_name>/<data>_gen_task<t>_<i>.png`，最终 `..._gen_all.png`。
- 指标：`results/<data>_<method>.pkl`（LL矩阵），`results/<data>_<method>_gen_class.pkl`（分类器 KL）。

## 2) 目录结构与调用关系

### 2.1 `dgm/` 文件树
```text
original_code/dgm/
├── README.md
├── config.py
├── exp.py
├── eval_kl.py
├── eval_ll.py
├── load_classifier.py
├── alg/
│   ├── eval_test_class.py
│   ├── eval_test_ll.py
│   ├── helper_functions.py
│   ├── onlinevi.py
│   ├── vae_ewc.py
│   ├── vae_laplace.py
│   └── vae_si.py
├── classifier/
│   ├── mnist.py
│   ├── notmnist.py
│   └── train_classifier.py
└── models/
    ├── bayesian_generator.py
    ├── encoder.py
    ├── encoder_no_shared.py
    ├── generator.py
    ├── import_data_mnist.py
    ├── mlp.py
    ├── mnist.py
    ├── notmnist.py
    ├── utils.py
    └── visualisation.py
```

### 2.2 调用链（训练）
`exp.py::main`  
→ `config.py::config`（任务/维度）  
→ `models/mnist.py::load_mnist`（按类别取数据）  
→ `models/(bayesian_)generator.py` + `models/encoder_no_shared.py`（建图）  
→ `alg/(onlinevi|vae_ewc|vae_laplace|vae_si).py::construct_optimizer`（训练函数闭包）  
→ `models/utils.py::init_variables/save_params`（初始化与任务后保存）  
→ `alg/eval_test_ll.py::construct_eval_func`（每 task 在已见任务上验证 LL）。

### 2.3 模块职责表（按重要度）
| 文件 | 职责 | 关键函数/类 | 输入→输出 |
|---|---|---|---|
| `exp.py` | 主实验编排（task 循环/训练/保存/可视化/valid LL） | `main` | `data_name,method,lbd` → checkpoints + figs + results |
| `alg/onlinevi.py` | VCL 核心：共享参数 KL 正则、先验更新、训练闭包 | `KL_param`, `construct_optimizer`, `update_shared_prior`, `update_q_sigma` | graph vars + prior dict → fit() |
| `models/bayesian_generator.py` | 贝叶斯生成器（参数有 μ/logσ，可采样） | `bayesian_mlp_layer`, `generator_shared`, `generator_head` | `z` → `x_recon` |
| `models/encoder_no_shared.py` | task-specific 编码器（无共享 encoder） | `encoder`, `sample_gaussian` | `x` → `mu_qz,log_sig_qz` |
| `alg/eval_test_ll.py` | test-LL 的 IS 估计 | `IS_estimate`, `construct_eval_func` | `X` → `(mean, ste)` |
| `alg/helper_functions.py` | 概率密度、KL、采样原语 | `KL`, `log_bernoulli_prob`, `sample_gaussian` | tensor → scalar/vector |
| `models/generator.py` | 非贝叶斯生成器（ewc/laplace/si/noreg） | `generator_shared/head` | `z` → `x_recon` |
| `alg/vae_ewc.py` | EWC 版 VAE 训练 | `compute_fisher`, `update_ewc_loss`, `construct_optimizer` | bound + fisher → fit() |
| `alg/vae_laplace.py` | Laplace propagation 版训练 | `init_fisher_accum`, `update_laplace_loss` | fisher 累积 + fit |
| `alg/vae_si.py` | SI 正则版训练 | `construct_optimizer`, `update_si_reg` | si_reg → fit |
| `eval_ll.py` | 离线评估 checkpoints 的 test-LL 和可视化 | `main` | saved ckpt → LL matrix |
| `eval_kl.py` | 生成样本的分类器不确定性评估 | `main` | saved ckpt + classifier → KL matrix |
| `alg/eval_test_class.py` | 分类器 KL 统计 | `KL_generated_images`, `construct_eval_func` | dec + cla → `(mean, ste)` |
| `models/utils.py` | TF1 变量初始化与参数序列化 | `init_variables`,`save_params`,`load_params` | session vars ↔ pkl |
| `load_classifier.py` / `classifier/train_classifier.py` | 分类器定义/训练/加载 | `load_model` | weights.h5 ↔ Keras model |

## 3) 生成任务的数据流与张量形状

### 3.1 任务定义与数据构造
- `labels = [[i] for i in xrange(10)]`，任务序列固定 0→9。每轮 `load_mnist(digits=labels[task-1])` 仅加载该类样本。
- 输入均为展平向量：`X_ph.shape = (batch_size, 784)`，不是 `1x28x28`。
- 每 task：`X_train` 再分 `0.9/0.1` 为 train/valid，`X_test` 保留用于最终评估。

### 3.2 网络结构与“共享/任务私有”
- Decoder（生成器）拆两段：
  - **task-specific head**：`generator_head(dimZ=50 -> dimH=500 -> 500)`，变量名含 `gen_<task>_head_l*`。
  - **shared trunk**：`generator_shared(500 -> 500 -> 784(sigmoid))`，变量名含 `gen_shared_l*`。
- Encoder 在 `exp.py` 里使用 `encoder_no_shared.encoder`：完全 task-specific，`enc_<task>_l*`，无共享 encoder。
  - 结构：`784 -> 500 -> 500 -> 500 -> 100`（最后 100 被 split 成 `mu/log_sig`，各 50 维）。

### 3.3 关键张量与参数组织
- `mu_qz/log_sig_qz`: `shape=(B,50)`。
- `z = mu + exp(log_sig)*eps`: `shape=(B,50)`（重参数化）。
- `mu_x = dec(z)`: `shape=(B,784)`，Bernoulli 参数。
- `onlinevi` 的 q(θ) 参数位于 `gen_shared`（与 head 也同构为 μ/logσ，但 KL 仅默认约束 shared）：
  - 每层 4 组参数：`mu_W, mu_b, log_sig_W, log_sig_b`。
- `shared_prior_params`: Python dict（numpy arrays），key 直接对齐变量名（如 `gen_shared_l0_mu_W:0`）。

## 4) 损失函数逐项展开（公式+代码对应）

### 4.1 VAE ELBO（每样本）
- 代码：`alg/(onlinevi|vae_ewc|vae_laplace|vae_si).py::lowerbound`。
- 统一形态：
  \[
  \mathcal{L}_{ELBO}(x)=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]-KL(q_\phi(z|x)||p(z))
  \]
- 实现细节：
  - `KL_z = KL(mu_qz, log_sig_qz, mu_pz=0, log_sig_pz=0)`。
  - `logp` 对 MNIST 用 `log_bernoulli_prob(x, mu_x)`。
  - `onlinevi` 允许 `K_mc` 次权重采样：循环采样 `z` 和 `theta`，平均 `logp`。

### 4.2 VCL 参数正则（onlinevi）
- 代码：`alg/onlinevi.py::KL_param` + `construct_optimizer`。
- 目标函数：
  \[
  \text{loss}= -\frac{1}{B}\sum \mathcal{L}_{ELBO}(x_b) + \frac{1}{N_t} KL(q_t(\theta_{shared})\|q_{t-1}(\theta_{shared}))
  \]
- `KL_param` 对每层 `W,b` 累加闭式高斯 KL（调用 `helper_functions.KL`，其定义是 `KL[p||q]`，传参顺序使其成为 `KL(q_t||prior)`）。
- 默认不正则 head（`regularise_headnet=False`）。因此任务记忆主要在 shared decoder 参数。

### 4.3 其他方法的附加项
- EWC/Laplace/SI 均在 `loss_total = -mean(bound) + reg/N_data(或同量纲)` 形式拼接。
- EWC/Laplace 的 Fisher 是按 batch 内每样本梯度平方近似并放大 `*N_data`。
- SI 使用论文式 `w_params` 累积路径积分近似重要度，再在下任务用二次惩罚。

### 4.4 Monte Carlo 与 mini-batch
- 训练时：
  - onlinevi: `K_mc=10`（默认）用于参数采样估计重构期望。
  - 其余法：通常单样本 `z` 采样。
- test-LL：`eval_test_ll.construct_eval_func(..., K=100)`（训练中 valid）或 `K=5000`（`eval_ll.py` 离线测试）。

## 5) Task 间“记忆”实现机制

### 5.1 每任务结束保存什么
- `save_params` 把 **当前图全部 trainable 变量** 存到 checkpoint（包含所有已建 task heads、shared decoder、已建 encoder）。
- 同时保存 `result_list` 到 `results/*.pkl`。

### 5.2 下一任务如何初始化（q_{t-1}→q_t）
- onlinevi：
  1. 任务 t 训练后 `update_shared_prior(sess, shared_prior_params)`：把当前 `gen_shared` 的 μ/logσ 数值复制到 Python 字典。  
  2. 下任务训练时 `KL_param(shared_prior_params, task)` 把该字典作为 prior。  
  3. `update_q_sigma(sess)` 将 `gen_shared` 的 `log_sig` 重置为 `-6`（打印文案写 -5，但代码是 -6）。
- 这对应论文里的 `q_{t-1}(θ)` 作为新任务先验（至少在 shared decoder 上）。

### 5.3 encoder 的处理
- 在 `exp.py` 使用 `encoder_no_shared`，每 task 新建 `enc_<task>`，不共享，不做跨任务贝叶斯正则；仅在该 task 的训练期优化。
- `eval_ll.py` 评估 task i 的 test-LL 时会实例化对应 `enc_i` 并载入 checkpoint 参数。

### 5.4 coreset
- `dgm/` 目录里**没有 coreset 逻辑**：无记忆样本缓冲、无回放集合维护函数、无 coreset 采样/合并训练步骤。

## 6) 评估与可视化复原

### 6.1 test-LL
- 实现：`alg/eval_test_ll.py::IS_estimate`。
- 方法：importance sampling（重复 K 次，log-sum-exp 稳定化）。
- 训练中 `K=100`（valid），离线评估 `K=5000`（test，`eval_ll.py`）。
- `sample_W=False` 时 decoder 用参数均值（不采样权重）评估。

### 6.2 classifier uncertainty
- 分类器来源：`classifier/train_classifier.py` 训练 MLP 后保存 `classifier/save/<data>_weights.h5`；`load_classifier.py` 载入。
- 指标实现：`alg/eval_test_class.py::KL_generated_images`，对 task t 的生成样本 `x_gen`，用固定 one-hot `y_true=t` 计算 `-log p_cla(y=t|x_gen)` 均值与方差（其实是 CE，不是对称 KL）。
- 评估脚本：`eval_kl.py` 在每个 checkpoint 上对所有已见 task 输出矩阵。

### 6.3 生成图（对应论文风格网格）
- 训练中：每个 task 后，对所有已见 head 各采样 100 张并单独保存；另外抽每个 head 的 1 张堆成 `10x10` 网格的累积图 `x_gen_all`。
- 离线评估：`eval_ll.py` 也构造 `x_gen_all` 并保存 `figs/visualisation/<data>_gen_all_<method>.png`。
- `plot_images` 实际通过 `reshape_and_tile_images(images, shape, n_rows=10)` 拼图；10 列固定，未见 task 的槽位填零图。

## 7) TF1/Keras1 迁移风险点清单（事实）

1. TF1 图模式：大量 `tf.placeholder` + `Session.run(feed_dict)`；PyTorch 需改为 eager 循环。  
2. `tf.all_variables / tf.initialize_variables` 已废弃；变量增量初始化逻辑需重写。  
3. 无 `variable_scope/reuse`，靠字符串命名区分任务头；迁移时要显式模块容器（`ModuleDict`）。  
4. Keras 1.2 + TF 混用：`keras.backend.set_session`, `learning_phase` feed。  
5. Python2 语法与 `cPickle`，Py3 需兼容改写。  
6. 参数保存不是 TF checkpoint，而是手工 `pkl(name->numpy)`；恢复依赖变量名完全一致。  
7. onlinevi 的 prior 作为 Python dict，不在计算图中；分任务状态管理要单独持久化。  
8. `update_q_sigma` 直接 in-graph `assign` 重置 logσ，训练稳定性依赖该 heuristic。  
9. MC 采样在图内 for-loop（Python 构图循环）；迁移要注意效率和随机数可复现。  
10. Fisher/SI 都是近似实现且按 shared vars 名称过滤，变量顺序耦合强。  
11. `batch_size_ph` 仅部分函数使用，存在“图里声明但用途弱”的遗留设计。  
12. `data_path = # TODO` 使脚本默认不可直接运行。

## “迁移到 PyTorch 时最需要的 20 个信息点”
1. 任务顺序固定 `0..9`，每任务单类。  
2. 输入是 `float32` 且归一化到 `[0,1]`，shape `[B,784]`。  
3. `dimZ=50`，decoder hidden `500`，batch `50`。  
4. Decoder: head(2层 relu) + shared(1层relu+sigmoid输出)。  
5. onlinevi 下 decoder 每层参数是 `mu/log_sig` 四元组（W,b）。  
6. shared decoder 是跨任务唯一持续对象。  
7. head decoder 按任务新增并保留。  
8. encoder 是**每任务独立**网络，不共享。  
9. ELBO 的重构项是 Bernoulli log-likelihood（非 BCE mean）。  
10. `KL_z` 用 `log_sig` 参数化（σ=exp(log_sig)）。  
11. onlinevi loss 有 `KL_theta/N_data` 缩放。  
12. onlinevi 每 batch 可做 `K_mc` 次 θ 采样平均。  
13. 任务间 prior 更新只作用 shared decoder（默认）。  
14. 每任务后 `log_sig` 被重置到 `-6`。  
15. checkpoint 是“所有可训练变量名→数组”的 pkl。  
16. 评估 LL 用 IS：训练中 K=100，离线 K=5000。  
17. `sample_W=False` 评估时固定权重均值。  
18. 分类器不确定性指标是 `-log p(y=t|x_gen)`。  
19. 图像网格是 10 列，未见任务位置零填充。  
20. 无 coreset / replay；记忆完全来自参数先验/正则。

## 附：短小关键代码片段（<=40行）

### A) 任务循环与模型拼装（`exp.py::main`）
```python
for task in xrange(1, N_task+1):
    X_train, X_test, _, _ = load_mnist(digits=labels[task-1], conv=False)
    dec = generator(generator_head(dimZ, dimH, n_layers_head, 'gen_%d' % task), dec_shared)
    enc = encoder(dimX, dimH, dimZ, n_layers_enc, 'enc_%d' % task)
    eval_func_list.append(construct_eval_func(X_ph, enc, dec, ll, batch_size_ph, K=100, sample_W=False))
    if method == 'onlinevi':
        fit = construct_optimizer(X_ph, enc, dec, ll, X_train.shape[0], batch_size_ph,
                                  shared_prior_params, task, K_mc)
    fit(sess, X_train, n_iter, lr)
    save_params(sess, filename, checkpoint)
    if method == 'onlinevi':
        shared_prior_params = update_shared_prior(sess, shared_prior_params)
        update_q_sigma(sess)
```

### B) onlinevi 总损失（`alg/onlinevi.py::construct_optimizer`）
```python
bound = tf.reduce_mean(lowerbound(X_ph, enc, dec, ll, K))
kl_theta = KL_param(shared_prior_params, task)
loss_total = -bound + kl_theta / N_data
opt = tf.train.AdamOptimizer(learning_rate=lr_ph).minimize(loss_total)
```

### C) IS test-LL（`alg/eval_test_ll.py::IS_estimate`）
```python
mu_qz, log_sig_qz = enc(x_rep)
z = sample_gaussian(mu_qz, log_sig_qz)
mu_x = dec(z, sampling=sample_W)
log_prior = log_gaussian_prob(z)
logq = log_gaussian_prob(z, mu_qz, log_sig_qz)
bound = tf.reshape(logp - (logq - log_prior), [K, N])
bound_max = tf.reduce_max(bound, 0)
log_norm = tf.log(tf.reduce_mean(tf.exp(bound - bound_max), 0))
test_ll = log_norm + bound_max
```

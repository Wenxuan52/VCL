# Variational Continual Learning

This repository was completed using `ChatGPT 5.2` and `Codex`, with references to [paper](https://arxiv.org/abs/1710.10628) and [code repository](https://github.com/nvcuong/variational-continual-learning).

## Train

```bash
python main.py mnist onlinevi
```

## Evaluation
To produce figure like Figure 6(e) in the paper, run

```bash
python eval_offline_IS_LL.py mnist onlinevi
```

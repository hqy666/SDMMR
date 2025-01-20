# Smooth Diffusion Model for Multimodal Recommendation

This is the PyTorch implementation for **SDMMR**.

<img src="./figures/model.png" style="zoom:100%;" />

## 📝 Environment

We develop our codes in the following environment:

- python==3.9.13
- numpy==1.23.1
- torch==1.11.0
- scipy==1.9.1

## 🎯 Experimental Results

Performance comparison of baselines on different datasets in terms of Recall@10, Recall@20, NDCG@10 and NDCG@20:

<img src="./figures/performance.png" style="zoom:100%;" />

## 🚀 How to run the codes

The command lines to train SDMMR on the three datasets are as below. The un-specified hyperparameters in the commands are set as default.

**! Due to dataset compression greater than 50M and double-blind policy, we provide training logs to demonstrate the authenticity of the performance. Each dataset will be published after acceptance of the paper.**  

- TikTok

```python
python Main.py --data tiktok --reg 1e-4 --ssl_reg 3e-2 --trans 1 --e_loss 0.1 --cl_method 1
```

- Baby

```python
python Main.py --data baby --reg 1e-5 --ssl_reg 3e-2 --keepRate 1 --e_loss 0.01
```

- Sports

```python
python Main.py --data sports --reg 1e-6 --ssl_reg 2e-2 --ris_lambda 0.1 --e_loss 0.5 --keepRate 1 --trans 1
```

## 👉 Code Structure

```
.
├── README.md
├── Main.py
├── Model.py
├── Params.py
├── DataHandler.py
├── Utils
│   ├── TimeLogger.py
│   └── Utils.py
├── figures
│   ├── model.png
│   ├── dataset.png
│   └── performance.png
└── Datasets
    └── tiktok
        ├── trnMat.pkl
        ├── tstMat.pkl
        ├── valMat.pkl
        ├── audio_feat.npy
        ├── image_feat.npy
        └── text_feat.npy
```

## 📚 Datasets

<img src="./figures/dataset.png" style="zoom:100%;" />

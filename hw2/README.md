# CS6180 HW2 — Neural Machine Translation

**Author:** Yiling Hu  
**Environment:** PyTorch + CUDA 12.4 (Local RTX 5070 Ti GPU)
**URL**: https://github.com/huyiling521/cs6180-genai

## Implemented Functions

| File | Function | Description |
|------|-----------|-------------|
| `utils.py` | `pad_sents(sents, pad_token)` | Pads each sentence to the same length with `<pad>`. |
| `model_embeddings.py` | `ModelEmbeddings.__init__()` | Initializes source & target embeddings with `nn.Embedding`, handling `padding_idx`. |
| `nmt_model.py` | `__init__()` | Defines post-embedding CNN, bidirectional LSTM encoder, LSTMCell decoder, attention & projection layers. |
| `nmt_model.py` | `encode()` | Runs the encoder on packed sequences and prepares decoder init states. |
| `nmt_model.py` | `decode()` | Iteratively decodes target sentences with attention. |
| `nmt_model.py` | `step()` | Executes one decoder step: computes attention weights, context vector, and combined output. |

All functions follow the assignment’s structure for encoder–decoder with attention.

## Environment Setup

Due to GCP GPU quota issues, training was performed **locally** on a **Windows machine (RTX 5070 Ti)** 

## Issues & Fixes

| Issue | Cause | Fix |
|--------|--------|-----|
| CUDA kernel error | PyTorch version incompatible with 5070 Ti | Upgraded to `torch==2.7.0.dev+cu124` |
| `_pickle.UnpicklingError` | `torch.load` default `weights_only=True` | Set `weights_only=False` |
| TensorBoard invalid arg | Used `--logdir = runs` | Corrected to `--logdir=runs` |

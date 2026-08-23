# Lab 21 — Submission Links

**Học viên**: Ngô Lưu Quốc Đạt · **MSSV**: 2A202602014

| | URL |
|---|---|
| GitHub repo | https://github.com/ngoluuquocdat/Day21-2A202602014-NgoLuuQuocDat |
| LoRA adapter (HuggingFace Hub) | `<điền sau khi push lên HF Hub — xem hướng dẫn bên dưới>` |

> Adapter `adapter_model.safetensors` nặng ~124 MB, vượt giới hạn 100 MB của GitHub nên
> **không** nằm trong repo — nó được đẩy lên HuggingFace Hub (bonus B5, +2đ). Bản đầy đủ
> cũng có trong `artifacts/adapters/correct/` trên máy local.

## Cách push adapter lên HF Hub (B5, +2đ)

```bash
pip install huggingface_hub
huggingface-cli login                       # dán token từ https://huggingface.co/settings/tokens
huggingface-cli upload ngoluuquocdat/lab21-qwen3.5-4b-triage-lora adapters/correct . --repo-type=model
```

Sau khi push xong, thay link ở bảng trên và cập nhật dòng B5 trong `submission/REPORT.md`.

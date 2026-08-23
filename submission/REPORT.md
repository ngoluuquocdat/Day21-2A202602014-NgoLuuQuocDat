# Lab 21 — Evaluation Report

**Họ tên**: Ngô Lưu Quốc Đạt  **MSSV**: 2A202602014  **Ngày**: 2026-08-22
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 (14.6 GB, Colab Free)`

> Mọi con số dưới đây khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định) |
| Train / eval | 250 train_seed · 50 eval_target + 15 eval_regression (seed 42) |
| `max_length` | 1024 (mặc định tier T4) — p95 đo được chỉ **98** token *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 30 |

**Template có giữ khối `<think>` không?** **Có** — `template_check.json` báo `open_tag_present: true`, `body_present: true`, kết luận *"reasoning preserved — safe to train on traces"*. Trên corpus này điều đó là vô hại: 250 câu trả lời train đều là JSON trần, và chat template của Qwen3.5 tự đóng khối `<think></think>` rỗng ngay trong prompt sinh, nên không còn đoạn suy luận nào lọt vào vùng được giám sát (xem mask proof §2).

**Về `max_length`**: p95 của corpus chỉ 98 token (tối đa 101), NB1 gợi ý `suggested_max_length=256`. Tier T4 lại ship mặc định `max_length=1024`. Độ lệch này **an toàn** vì 1024 ≫ p95=98 → **không mẫu nào bị cắt**; cái giá duy nhất là padding thừa, nhưng vì chuỗi thực rất ngắn nên chi phí compute không đáng kể. Nếu tối ưu, đặt 128 hoặc 256 là đủ và nhanh hơn.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

`supervised_fraction = 0.41 < 0.95` → loss **không** tính trên prompt. Đoạn được tính loss (supervised span):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần bị mask (system + user + mở `<think>`) **không** vào loss — đúng như thiết kế `assistant-only`.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train, n=50)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.00 | 3232.5 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.00 | 1047.8 |
| (c) LoRA fine-tune | **0.975** | **0.500** | 1.00 | 1375.9 |

**(b) có thật sự mạnh hơn (a) không?** **Có** — target 0.000 → 0.765, format 0.00 → 1.00. Gatekeeper xác nhận `baseline (b) beats (a)`. Tôi **không** sửa `OPTIMIZED_PROMPT`: `optimized_prompt_sha` khớp bản gốc (`verify.py` báo *"baseline (b) prompt unmodified"*). Nghĩa là (c) phải vượt một mốc **thật sự khó** (0.765), không phải một prompt bị làm yếu để ăn gian.

> Đọc kỹ hàng (a): với prompt ngây thơ, model **không parse được JSON nào** (format 0.0, target 0.0). Chỉ riêng việc dựng đúng prompt đã kéo target từ 0 lên 0.765 — bài học lớn đầu tiên: *prompt engineering đúng cách là baseline bắt buộc phải đánh bại trước khi nói fine-tune "thắng".*

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | steps | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6262 | **0.975** | 30 | 12.01 |
| `attn_only` | q,v (attn) | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5366 | 0.970 | 30 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | **0.000** | 30 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 30 | 7.09 |

Ngân sách tham số khớp: `correct` 32,464,896 vs `attn_only` 32,456,704 → lệch **0.025%** (< 5%, gatekeeper xác nhận). Cả 4 run cùng **30 step**.

**4.1 — `attn_only` (cùng số tham số với `correct`): thắng, thua, hay hoà? Rank vs vị trí?**
Trên target, `attn_only` đạt **0.970** so với `correct` **0.975** — gần như **hoà** (chênh 0.005). Nhưng chú ý mâu thuẫn: `attn_only` có **train loss thấp hơn** (0.5366 < 0.6262) mà target lại **không cao hơn**. Nếu chấm bằng train loss, tôi đã kết luận sai rằng `attn_only` tốt hơn. Điều này nói rằng: khi **ngân sách tham số đã bằng nhau**, việc dồn rank vào riêng attention (q,v ở r=283) so với trải mỏng khắp text-linear (r=16) **gần như không tạo khác biệt trên bài toán này**. Rank cao không phải đòn bẩy; **vị trí gắn adapter** cũng chỉ là đòn bẩy rất nhỏ ở đây. Đòn bẩy thật nằm ở chỗ khác (xem 4.2).

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss? Kết luận sai gì nếu chỉ nhìn loss?**
`wrong_lr` dùng LR 1e-5 (thang full-fine-tune) thay vì 1e-4 (thang LoRA), chỉ khác **một biến duy nhất**. Train loss của nó **kẹt ở 1.57** — cao gấp ~2.5 lần `correct` (0.63) — vì LR quá nhỏ để adapter kịp học trong 30 step; loss gần như phẳng. Hệ quả trên target: **0.000** — sập hoàn toàn, model không xuất được JSON hợp lệ. Đây là đòn bẩy **lớn nhất** đo được trong cả lab: **learning rate**. Nếu ai đó chỉ nhìn train loss của `correct` (0.63) mà không biết dải LR, họ sẽ không hiểu vì sao cùng cấu hình đó với LR nhỏ 10× lại cho **điểm 0**.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?**
`qlora` (4-bit) đỉnh VRAM **7.09 GB** so với **12.01 GB** của `correct` → tiết kiệm **4.92 GB (~41%)**. Nhưng trả giá kép: target tụt **0.975 → 0.940** (mất ~3.5 điểm chính xác do sai số lượng tử hoá), **và** latency **chậm hơn** (1765 ms vs 1376 ms, do phải giải lượng tử khi sinh). Số đo của tôi **ủng hộ** khuyến nghị "không dùng QLoRA cho dòng Qwen3.5" (deck §12): ở 4B, bf16 LoRA vẫn vừa T4 (12 GB < 14.6 GB), nên đánh đổi 41% VRAM để mất cả độ chính xác lẫn tốc độ là **không đáng** — QLoRA chỉ nên dùng khi model lớn tới mức bf16 không vừa card.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.210` · `regression Δ = −0.258` · `valid_trace_rate = 0.00`

**Diễn giải.** Cổng hồi quy **FAIL**, và đây là kết quả có giá trị nhất tôi đo được. Bản fine-tune **thắng rõ trên nhiệm vụ đích**: target 0.765 → 0.975 (+0.210 so với prompt tối ưu đã rất mạnh), format giữ 1.0, latency chấp nhận được. Nhưng nó **phá huỷ năng lực phổ thông**: regression rớt từ 0.758 xuống **0.500** (−0.258), vượt xa ngưỡng dung sai 0.020. Đây là **quên thảm hoạ (catastrophic forgetting)** kinh điển (deck §14.3): khi ép model vào một phân phối hẹp (JSON 4 trường) trong 30 step mà **không trộn dữ liệu replay phổ thông**, nó đánh đổi kiến thức tổng quát để lấy độ khớp nhiệm vụ. Kết luận thực dụng: **KHÔNG nên deploy bản này nguyên trạng** — một chatbot CSKH vẫn cần trả lời được câu hỏi ngoài triage; mất một nửa năng lực phổ thông là cái giá quá đắt cho +0.21 điểm target. Hướng sửa: trộn 1–5% dữ liệu phổ thông vào tập train rồi đo lại regression. (`valid_trace_rate = 0.0` vì corpus toàn JSON trần, không có reasoning trace để đo — đúng như kỳ vọng, không phải lỗi.)

---

## 6. Định tính — có cả ca THUA

*Lưu ý: NB5 chỉ dump dự đoán của bản fine-tune (c) theo từng mẫu; cột (b) per-item không được lưu, nên tôi so (c) với **nhãn đúng** và đối chiếu tổng thể (b)=0.765 vs (c)=0.975. `ft_pred` lưu dạng preview cắt ~85 ký tự nên trường `sentiment` thường bị cắt; suy ra field sai từ 3 trường hiển thị + `ft_score`.*

| # | Ticket (rút gọn) | Nhãn đúng (field then chốt) | (c) fine-tune | Kết quả |
|---|---|---|---|---|
| 1 | "…ốp lưng điện thoại DH734695. Giá bao nhiêu." | intent=`hoi_thong_tin` | intent=`hoi_thong_tin` ✓ (score 1.0) | ✅ FT thắng |
| 2 | "…chuột không dây VN232232…" (đổi/trả, khẩn cao) | urgency=`cao`, intent=`doi_tra` | khớp cả 4 trường (score 1.0) | ✅ FT thắng |
| 3 | "…bình giữ nhiệt VN804124. Chưa thấy tiền." | urgency=`thap` | urgency=`trung_binh` ✗ (score 0.75) | ❌ **FT thua** |
| 4 | "…áo khoác gió VN613097. Bị lỗi. Khi nào tiện." | urgency=`thap` | urgency=`trung_binh` ✗ (score 0.75) | ❌ **FT thua** |
| 5 | "…nồi chiên không dầu VN949966. Hoàn tiền." | urgency=`thap` | urgency=`trung_binh` ✗ (score 0.75) | ❌ **FT thua** |

**Mẫu chung ở các ca thua:** cả 5 mẫu điểm thấp nhất (0.75) đều sai **đúng một trường: `urgency`**, và luôn theo cùng một chiều — nhãn đúng là `thap` nhưng FT đoán `trung_binh`. Các ticket này có tín hiệu "không gấp" ("Khi nào tiện", "Chưa thấy tiền" nhưng giọng bình thản). Fine-tune đã **sập về lớp đa số `trung_binh`** và mất khả năng phân biệt mức khẩn thấp — đúng kiểu lỗi mà một tập train nhỏ, mất cân bằng lớp dễ gây ra. Đây cũng là gợi ý dữ liệu: cần thêm mẫu `urgency=thap` hoặc cân bằng lại lớp.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không nên deploy** bản fine-tune này nguyên trạng, dù nó thắng prompt tối ưu +0.21 điểm target. Lý do là cổng hồi quy bốn nhóm phơi bày một đánh đổi mà chỉ nhìn điểm target sẽ bỏ sót: model đã **quên thảm hoạ** năng lực phổ thông (regression 0.758 → 0.5). Với một trợ lý CSKH thật, mất một nửa khả năng trả lời ngoài phạm vi triage là rủi ro không chấp nhận được. Về **đòn bẩy thật sự**: thí nghiệm đối chứng cùng ngân sách cho câu trả lời rõ ràng — **learning rate là đòn bẩy lớn nhất** (`wrong_lr` với LR nhỏ 10× rớt từ 0.975 xuống 0.000), trong khi **vị trí adapter và rank gần như không tạo khác biệt** khi ngân sách tham số đã khớp (`attn_only` 0.970 ≈ `correct` 0.975, dù train loss của nó còn thấp hơn). QLoRA thì lỗ kép (mất 41% ưu thế… thực ra tiết kiệm 41% VRAM nhưng mất cả độ chính xác lẫn tốc độ), nên với 4B trên T4 không có lý do dùng nó. Bài học tổng: **chấm bằng chỉ số thay thế (train loss, hoặc chỉ target) sẽ dẫn tới kết luận sai**; chỉ có cổng hồi quy nhiều nhóm mới cho phán quyết trung thực. Điều quý nhất lab dạy tôi không phải "làm sao để thắng" mà "làm sao để **biết** mình có thắng hay không".

**Ba điều tôi học được** (cụ thể):
1. **Train loss là bẫy.** `attn_only` có train loss thấp hơn `correct` nhưng target không cao hơn; nếu xếp hạng bằng loss tôi đã chọn sai cấu hình. Phải chấm bằng chỉ số nhiệm vụ trên tập đóng băng.
2. **Fine-tune có thể thắng nhiệm vụ mà vẫn là quyết định sai.** +0.21 target đi kèm −0.258 regression — chỉ cổng bốn nhóm mới lộ ra; một pipeline chỉ đo accuracy đã "ăn mừng" nhầm.
3. **Learning rate là đòn bẩy át tất cả.** Cùng một cấu hình, đổi đúng LR từ 1e-4 sang 1e-5 biến điểm 0.975 thành 0.000. Vị trí/rank chỉ là chi tiết phụ khi ngân sách đã khớp.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 3% dữ liệu phổ thông (general instruction) vào tập train và chạy lại NB3→NB5 để xem regression có phục hồi qua 0.02 tolerance mà vẫn giữ được target ~0.95 không — tức là biến một verdict FAIL thành PASS bằng đúng phương thuốc §14.3, và đo chính xác cái giá dữ liệu phải trả.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:

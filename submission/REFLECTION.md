# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Bản fine-tune **thắng rõ trên nhiệm vụ** (target 0.765 → 0.975) nhưng cổng hồi quy lại **FAIL** — nó quên mất một nửa kiến thức phổ thông (regression 0.758 → 0.5). Tôi đã đinh ninh "điểm target lên là thành công", nên việc một model giỏi hơn ở việc được giao lại là quyết định deploy **sai** khiến tôi bất ngờ nhất. Cú sốc thứ hai: chỉ đổi learning rate từ 1e-4 xuống 1e-5 mà điểm rơi thẳng từ 0.975 xuống 0.000.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Không phải ở phần train như tôi tưởng, mà ở **hạ tầng**: GPU laptop của tôi chỉ 4 GB nên không train nổi tier nào tại chỗ, phải chuyển sang Colab T4; rồi lần chạy đầu lỡ để `EVAL_LIMIT=8` (smoke) làm hai run `correct`/`attn_only` hoà nhau, phải chạy lại full 50 mẫu; cuối cùng là lỗi xuống dòng CRLF của Windows làm `verify.py` báo sai checksum. Phần "học máy" thật ra chạy trơn; phần "kỹ thuật vận hành" mới ngốn thời gian.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Tôi từng tin (a) rank càng cao và gắn adapter vào attention thì càng mạnh, và (b) train loss thấp hơn nghĩa là model tốt hơn. Cả hai đều sai trong lab này: khi **ngân sách tham số đã khớp**, `attn_only` (r=283) và `correct` (r=16) gần như hoà, dù `attn_only` có train loss **thấp hơn** mà điểm target không cao hơn. Đòn bẩy thật là **learning rate** và **chất lượng/độ cân bằng dữ liệu**, không phải rank hay vị trí.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi dùng AI để gỡ lỗi `No module named labkit` trên Colab, tìm ra nguyên nhân checksum sai là do CRLF của Windows, diễn giải verdict FAIL, và dựng khung báo cáo. Nó không "sai" ở phép tính, nhưng có một chỗ phải kiểm lại: khi phân tích ca thua, nó **suy ra** trường bị sai là `urgency` từ `ft_pred` đã bị cắt ngắn trong `qualitative.json` — đó là suy luận gián tiếp, nên tôi đối chiếu lại với nhãn đúng trong `eval_target.jsonl` để xác nhận. Bài học: AI tăng tốc gỡ lỗi và phân tích rất tốt, nhưng **mọi con số cuối cùng tôi vẫn phải tự kiểm chứng với file trong `results/`**.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Đóng băng một tập eval **trước khi** train, và tập đó phải gồm cả nhiệm vụ đích **lẫn** một bộ câu hỏi phổ thông để bắt quên thảm hoạ — vì đó chính là thứ suýt qua mặt tôi hôm nay. Song song, đo baseline prompt tối ưu trước để biết fine-tune có thật sự đáng công so với chỉ prompt cho tử tế. Chỉ khi có mốc trung thực đó tôi mới bắt đầu train, và sẽ trộn sẵn 1–5% dữ liệu replay để phòng regression ngay từ đầu.

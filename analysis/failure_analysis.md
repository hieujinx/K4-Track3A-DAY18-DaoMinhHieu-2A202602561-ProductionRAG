# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Đào Minh Hiếu
**Khóa:** K4 - Track 3A
**Ngày phân tích:** 05/10/2026

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8333 | 0.8120 | -0.0213 |
| Answer Relevancy | 0.7548 | 0.7247 | -0.0301 |
| Context Precision | 0.7750 | 0.6833 | -0.0917 |
| Context Recall | 0.8250 | 0.6833 | -0.1417 |

Production không cải thiện so với baseline trong lần chạy này. Nguyên nhân chính là child chunk 256 ký tự bị cắt giữa bảng/câu, pipeline chưa mở rộng child thành parent khi tạo câu trả lời, enrichment làm đoạn dài hơn nhưng không phục hồi phần bị cắt, và reranker chưa ưu tiên phiên bản tài liệu mới.

## Bottom-5 Failures

### #1 — Phê duyệt thiết bị 55 triệu (điểm trung bình 0.2500)

- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** “Không tìm thấy.”
- **Model trả lời đúng không?** Không. Câu trả lời không cung cấp người phê duyệt.
- **Context có chứa đáp án không?** Context đầu tiên tìm đúng bảng thẩm quyền mua sắm, nhưng bị cắt ngay sau ô `Trên 50.000.000 VNĐ |`; ô “Tổng Giám đốc (CEO)” nằm ngoài child chunk nên không đến được LLM.
- **Query có cần viết lại không?** Không. Câu hỏi rõ giá trị, hành động và thông tin cần tìm.
- **Worst metric:** `answer_relevancy = 0.0`, `context_precision = 0.0`, `context_recall = 0.0`.
- **Error Tree:** Output sai → context chỉ đúng một phần → query tốt → lỗi ở M1/pipeline parent-child.
- **Root cause:** Hierarchical chunking cắt bảng theo giới hạn ký tự và pipeline chỉ lưu/gửi child, chưa truy ngược parent. Hai context còn lại nói về tạm ứng và hoàn ứng nên gây nhiễu.
- **Suggested fix:** M1 phải giữ nguyên bảng hoặc tạo overlap; lưu parent store và sau khi retrieve child phải mở rộng sang parent trước khi rerank/generate. Có thể thêm metadata `section=mua_sam/tham_quyen` để lọc M2.

### #2 — Nghỉ không lương 20 ngày (điểm trung bình 0.3750)

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Giám đốc điều hành (CEO); nghỉ trên 14 ngày còn phải tự đóng phần bảo hiểm của nhân viên.
- **Got:** “Không tìm thấy.”
- **Model trả lời đúng không?** Không.
- **Context có chứa đáp án không?** Gần đủ nhưng bị cắt giữa cụm `cần phê duyệt của **Giám đốc điều...`; phần “hành (CEO)” và lưu ý bảo hiểm không nằm trong context gửi LLM.
- **Query có cần viết lại không?** Không; mốc 20 ngày ánh xạ rõ vào khoảng 16–30 ngày.
- **Worst metric:** `faithfulness = 0.0`, `answer_relevancy = 0.0`; `context_precision ≈ 1.0` cho thấy retriever tìm đúng tài liệu nhưng context bị thiếu phần kết luận.
- **Error Tree:** Output sai → context đúng tài liệu nhưng thiếu đuôi câu → query tốt → lỗi ở M1 và bước child-to-parent expansion.
- **Root cause:** Hard split 256 ký tự làm vỡ câu. Hai kết quả tiếp theo là phép năm 2023/2024, không giúp trả lời phép không lương.
- **Suggested fix:** Cắt con theo biên câu với overlap, hoặc gửi parent 2048 ký tự tương ứng. M3 nên ưu tiên cùng loại nghỉ phép và giảm điểm các chunk “phép năm” khi query chứa “không lương”.

### #3 — Phạt tạm ứng quá hạn (điểm trung bình 0.4546)

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; 2%/tháng trên 15 triệu là 300.000 VNĐ/tháng, pro-rata khoảng 50.000 VNĐ cho 5 ngày.
- **Got:** Nêu đúng mức 2%/tháng và cơ chế khấu trừ nhưng không tính ra 300.000 VNĐ/tháng hoặc khoảng 50.000 VNĐ cho 5 ngày.
- **Model trả lời đúng không?** Đúng một phần, chưa trả lời trực tiếp “bao nhiêu”.
- **Context có chứa đáp án không?** Có đủ hạn 15 ngày, mức 2%/tháng và số tiền 15 triệu; LLM có thể tự tính từ context.
- **Query có cần viết lại không?** Không. Đây là câu hỏi nhiều bước nhưng đã nêu đủ dữ kiện.
- **Worst metric:** `context_precision = 0.0`; faithfulness chỉ `0.4` vì phần diễn giải “20 ngày vượt quá 15 ngày quy định” chưa chuyển thành phép tính được kiểm chứng.
- **Error Tree:** Output thiếu phép tính → context đủ → query tốt → lỗi ở prompt/generation.
- **Root cause:** Prompt chỉ yêu cầu trả lời dựa trên context, không yêu cầu trình bày phép tính và kết quả số cuối cùng.
- **Suggested fix:** Sửa `run_query()` để yêu cầu trả lời trực tiếp, thực hiện phép tính từng bước khi câu hỏi có số liệu, và kiểm tra đơn vị/pro-rata. Có thể thêm calculator tool ở tầng generation.

### #4 — Senior 9 năm: phép và lương (điểm trung bình 0.4825)

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép theo v2024 và lương Senior (P3–P4) 20–35 triệu VNĐ/tháng.
- **Got:** Trả đúng 18 ngày phép nhưng “Không tìm thấy” mức lương.
- **Model trả lời đúng không?** Đúng một nửa.
- **Context có chứa đáp án không?** Có đủ thông tin phép năm; không có chunk bảng lương. Hai trong ba context lại là chính sách phép năm 2023 cũ.
- **Query có cần viết lại không?** Người dùng không cần viết lại; hệ thống nên tự phân rã câu hỏi thành hai sub-query “phép năm theo thâm niên” và “dải lương Senior”.
- **Worst metric:** `context_precision = 0.0`, `context_recall = 0.0`.
- **Error Tree:** Output thiếu một vế → context thiếu tài liệu lương → query multi-hop hợp lệ → lỗi ở M2 retrieval/query decomposition và M3 diversity.
- **Root cause:** Top-3 bị chiếm bởi nhiều chunk cùng chủ đề phép năm, gồm cả bản cũ, nên không còn slot cho bảng lương.
- **Suggested fix:** Tách multi-hop query, retrieve từng sub-query rồi fusion; áp dụng source diversity/MMR trước rerank và tăng candidate/top context có kiểm soát.

### #5 — Mốc thâm niên cộng phép (điểm trung bình 0.4851)

- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Chính sách hiện hành v2024: từ 3 năm, cộng 1 ngày mỗi 3 năm; v2023 cũ là 5 năm.
- **Got:** “Từ 5 năm trở lên (hoặc từ 3 năm trở lên tùy theo đoạn quy định)”.
- **Model trả lời đúng không?** Không đủ chính xác vì không chọn chính sách hiện hành và đặt bản cũ trước.
- **Context có chứa đáp án không?** Có cả v2023 và v2024, nhưng bản 2023 được rerank lên đầu.
- **Query có cần viết lại không?** Có thể thêm “theo chính sách hiện hành”, nhưng trong hệ thống doanh nghiệp mặc định phải ưu tiên văn bản mới nhất nên không nên đẩy trách nhiệm này cho người dùng.
- **Worst metric:** `context_precision = 0.0`, `context_recall = 0.0`; faithfulness và relevancy cao vì câu trả lời bám context nhưng chọn sai tính hiệu lực.
- **Error Tree:** Output có dữ kiện nhưng không giải quyết xung đột → context chứa cả cũ/mới → query chấp nhận được → lỗi ở metadata/version-aware retrieval và generation policy.
- **Root cause:** Không có quy tắc ưu tiên `v2024`/ngày hiệu lực trong M2–M3 và system prompt không yêu cầu chọn phiên bản mới nhất.
- **Suggested fix:** Trích xuất `effective_year`, `version`, `status`; boost tài liệu mới, hạ điểm bản superseded và thêm prompt “nếu có xung đột, dùng văn bản mới nhất và nêu bản cũ chỉ để đối chiếu”.

## Case Study (cho presentation)

**Question chọn phân tích:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

**Error Tree walkthrough:**

1. Output đúng? → Không, trả “Không tìm thấy”.
2. Context đúng? → Retriever tìm đúng bảng, nhưng child chunk cắt mất chính ô chứa CEO.
3. Query rewrite OK? → Query đã rõ, rewrite không giải quyết việc chunk bị cắt.
4. Fix ở bước → M1 giữ cấu trúc bảng + pipeline mở rộng child về parent; M2 lọc theo section mua sắm.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Thêm parent store và child-to-parent expansion trước reranker.
- Không hard-split bảng Markdown; bổ sung overlap theo câu.
- Thêm metadata phiên bản/năm hiệu lực và source diversity.
- Chạy ablation: raw child vs parent-expanded vs structure-aware, đo lại 4 RAGAS metrics trên cùng 20 câu.

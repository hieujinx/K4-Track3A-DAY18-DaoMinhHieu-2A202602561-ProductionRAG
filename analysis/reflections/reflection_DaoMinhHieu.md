# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Đào Minh Hiếu
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Tách câu bằng regex, encode với `all-MiniLM-L6-v2`, tính cosine giữa hai câu liên tiếp và ngắt khi dưới threshold 0.85. Cách này giữ biên câu tốt hơn hard split nhưng cần model embedding và có chi phí khởi tạo. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Pipeline tạo 117 child chunks từ 26 tài liệu, so với 57 basic chunks của baseline. Child giúp retrieval chính xác, nhưng kết quả tích hợp cho thấy phải mở rộng lại parent khi generate; nếu chỉ gửi child 256 ký tự thì bảng/câu có thể bị cắt mất đáp án. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Regex tiêu đề Markdown cấp 1–3 giữ `section` trong metadata. Kết quả Bottom-5 cho thấy cần dùng strategy này trực tiếp cho bảng hoặc kết hợp với hierarchical thay vì hard split mọi đoạn. |
| Vietnamese BM25 | M2 | `segment_vietnamese()`, `BM25Search` | `underthesea` tách từ và thay `_` bằng khoảng trắng để query “nghỉ phép” khớp tài liệu. BM25 bổ sung khả năng bắt số tiền, số ngày và thuật ngữ chính xác. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF cộng `1/(k+rank+1)` từ lexical và dense mà không cần chuẩn hóa hai thang điểm. Production vẫn bị trùng nhiều chunk phép năm trong top-3, nên bước tiếp theo cần source diversity/MMR. |
| Dense retrieval | M2 | `DenseSearch.index()`, `DenseSearch.search()` | Dùng `BAAI/bge-m3` 1024 chiều và Qdrant `query_points()`. Qdrant remote timeout 2 giây đã gây lỗi ở query thứ hai; tăng timeout lên 30 giây giúp tái sử dụng đủ 117 points đã index. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` chấm trực tiếp cặp query–document và lấy top-3. Model được cache theo tên để tránh nạp lại, nhưng CPU latency còn cao; reranker cũng chưa có tín hiệu version/freshness để giải quyết v2023–v2024. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Đã chấm đủ 20 câu × 4 metrics. Baseline: 0.8333/0.7548/0.7750/0.8250; Production: 0.8120/0.7247/0.6833/0.6833. Kết quả âm cho thấy kiến trúc “advanced” không tự động tốt hơn nếu parent expansion và version policy chưa đúng. |
| Diagnostic Tree | M4 | `failure_analysis()` | Tính trung bình bốn metric, tìm metric thấp nhất rồi ánh xạ diagnosis/fix. Phân tích thủ công Bottom-5 bổ sung bằng chứng context: hai đáp án bị cắt đúng giữa bảng/câu, một câu thiếu phép tính, một câu multi-hop thiếu nguồn lương và một câu xung đột phiên bản. |
| Contextual embeddings / HyQA | M5 | `_enrich_single_call()`, `contextual_prepend()` | 117 chunks được enrich thành công trong 588,2 giây bằng một call/chunk. Context prepend giúp mô tả nguồn/chủ đề nhưng làm text dài hơn; nếu child gốc đã mất đuôi đáp án thì enrichment không thể phục hồi thông tin đó. |

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. API key và provider không tương thích

- **Exact error:** `AuthenticationError: Your API key has been invalidated.`
- **Nguyên nhân:** Code ban đầu dùng OpenAI endpoint/model `gpt-4o-mini`, trong khi key mới là Google Gemini; key OpenAI cũ đã bị vô hiệu hóa.
- **Cách debug và sửa:** Chỉ kiểm tra tên biến môi trường, không in secret; xác minh model bằng `models.list()`. Dùng OpenAI-compatible Gemini endpoint `https://generativelanguage.googleapis.com/v1beta/openai/`, model `gemini-3.5-flash-lite` và `gemini-embedding-001`.

### 2. Model cũ không còn cho tài khoản mới

- **Exact error:** `404 ... gemini-2.5-flash-lite is no longer available to new users`.
- **Nguyên nhân:** Model vẫn xuất hiện trong danh sách nhưng API chặn tài khoản mới.
- **Cách debug và sửa:** Chạy một chat smoke test; chuyển sang model được API đề xuất là `gemini-3.5-flash-lite`, sau đó xác nhận chat trả `OK` và embedding có 3072 chiều.

### 3. RAGAS yêu cầu nhiều candidates nhưng Gemini không hỗ trợ

- **Exact error:** `400 Multiple candidates is not enabled for this model`.
- **Nguyên nhân:** `AnswerRelevancy` của RAGAS mặc định `strictness=3`, tương đương yêu cầu nhiều candidates trong một call.
- **Cách debug và sửa:** Khởi tạo `AnswerRelevancy(strictness=1)`. Smoke test một mẫu sau sửa cho cả bốn metric xấp xỉ 1.0.

### 4. Vượt quota Google free tier

- **Exact error:** `429 Quota exceeded ... limit: 15 ... generate_content_free_tier_requests`.
- **Nguyên nhân:** Baseline generation và 80 RAGAS jobs tạo burst lớn hơn 15 request/phút.
- **Cách debug và sửa:** Tạo `src/llm_provider.py` với `InMemoryRateLimiter` dùng chung cho generation, enrichment và RAGAS ở mức 12 request/phút. Lượt chạy cuối hoàn tất cả baseline và production RAGAS mà không có 429.

### 5. Qdrant timeout sau khi đã enrichment

- **Exact error:** `qdrant_client.http.exceptions.ResponseHandlingException: timed out`.
- **Nguyên nhân:** Remote Qdrant client được đặt timeout 2 giây. Query đầu chạy được nhưng query thứ hai timeout.
- **Cách debug và sửa:** Tăng timeout lên 30 giây, kiểm tra collection còn đủ 117 points và payload, rồi phục hồi production evaluation từ collection thay vì gọi lại 117 enrichment requests.

### 6. Kết quả Production thấp hơn Baseline

- **Quan sát:** Δ Faithfulness -0.0213, Answer Relevancy -0.0301, Context Precision -0.0917, Context Recall -0.1417.
- **Nguyên nhân gốc:** Pipeline index child nhưng không lưu/expand parent khi generate; hard split làm vỡ bảng/câu; top-3 thiếu diversity cho multi-hop; chưa có version-aware ranking; prompt chưa yêu cầu tính toán trực tiếp.
- **Bài học:** Cần đánh giá end-to-end và làm ablation. Thêm nhiều kỹ thuật không đảm bảo cải thiện nếu contract giữa các module (retrieve child → return parent) chưa được thực thi đầy đủ.

## Phần 3: Action Plan cho Project cá nhân

### Project: Trợ lý tra cứu chính sách doanh nghiệp có quản lý phiên bản

#### 1. Hiện trạng

- **Pipeline hiện tại:** Markdown/PDF → hierarchical child chunks → BM25 + BGE-M3/Qdrant → RRF → BGE reranker → Gemini answer → RAGAS.
- **Vấn đề/Bottlenecks:** Mất ngữ cảnh parent, bảng bị cắt, top-3 trùng nguồn, nhầm phiên bản cũ, CPU reranking chậm và quota LLM thấp.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Dùng structure-aware cho Markdown/bảng, hierarchical cho prose; child 256 có overlap theo câu và parent 2048 được lưu trong parent store.
2. **Search retrieval:** Hybrid BM25 + BGE-M3 + RRF; thêm query decomposition cho câu multi-hop, metadata filter theo loại chính sách và MMR/source diversity.
3. **Parent expansion:** Retrieve child để lấy precision, ánh xạ `parent_id` rồi gửi parent đầy đủ vào reranker/generator; loại bỏ parent trùng lặp.
4. **Version policy:** Trích xuất `effective_date`, `version`, `status`; boost văn bản mới và đánh dấu bản superseded. Prompt phải ưu tiên chính sách còn hiệu lực.
5. **Reranking:** Giữ `BAAI/bge-reranker-v2-m3`, benchmark CPU/GPU; đưa freshness, source và section vào feature/rule sau cross-encoder.
6. **Evaluation:** Giữ cố định bộ 20 câu và bổ sung riêng nhóm table-boundary, multi-hop, numeric calculation, version conflict. Theo dõi cả quality, p95 latency, token/API cost và fallback rate.
7. **Enrichment:** Chỉ contextual prepend cho chunk thiếu tiêu đề; cache theo hash nội dung, chạy batch offline và không enrich phần đã có heading rõ để tránh pha loãng từ khóa.

#### 3. Timeline triển khai

- **Tuần 1:** Xây parent store, structure-aware table chunker, sentence overlap và metadata phiên bản; viết unit/integration tests cho bốn nhóm failure.
- **Tuần 2:** Thêm query decomposition + MMR, tối ưu reranker latency, chạy ablation và RAGAS; chỉ rollout nếu Production vượt baseline ở context precision/recall và không tăng p95 quá ngưỡng cho phép.

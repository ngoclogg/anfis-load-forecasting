# ANFIS Load Forecasting

Repo này là workspace của dự án ANFIS. Giữ thiết lập nhỏ, rõ ràng và dễ mở rộng.

## Bề Mặt Chính

- Gốc kỹ năng Codex/Gemini: `.agents/skills/`
- Cấu hình dự án Codex: `.codex/config.toml`, `.codex/AGENTS.md`, `.codex/anfis-skill-root.toml`
- Manifest plugin Codex: `.codex-plugin/plugin.json`
- Marketplace plugin Codex: `.agents/plugins/marketplace.json`
- Điểm vào gốc Gemini: `.gemini/GEMINI.md`
- Hướng dẫn thiết lập và mở rộng Codex: `.agents/skills/anfis-guide/SKILL.md`
- Kỹ năng điều phối Codex: `.agents/skills/anfis-project-agents/SKILL.md`

## Quy Tắc Làm Việc

1. Đọc `docs/prd.html`, `docs/spec.html`, `docs/review.html` và `docs/review-spec.html` trước khi thay đổi quy trình ANFIS.
2. Ưu tiên chỉnh sửa nhỏ, đúng mục tiêu thay vì viết lại.
3. Bảo toàn LaTeX, toán học, bảng và các phần báo cáo hiện có trừ khi người dùng yêu cầu thay đổi rõ ràng.
4. Giữ đường dẫn tương đối với `anfis-load-forecasting/`.
5. Không mã hóa cứng đường dẫn tuyệt đối theo máy hoặc đường dẫn Kaggle.
6. Tóm tắt giả định khi phạm vi dữ liệu, định nghĩa biến mục tiêu hoặc quy tắc đánh giá còn mơ hồ.
7. Giữ nội dung kỹ năng tái sử dụng trong `.agents/skills/`.
8. Giữ cấu hình Codex trong `.codex/` và `.codex-plugin/`; giữ cấu hình Gemini trong `.gemini/`.
9. Luôn đọc và ghi file tiếng Việt của dự án bằng UTF-8. Trong PowerShell, đặt `$OutputEncoding` và `[Console]::OutputEncoding` thành UTF-8 và dùng `Get-Content -Encoding UTF8`; với lệnh Python trên Windows, đặt `PYTHONUTF8=1`; khi script phải ghi text, dùng UTF-8 rõ ràng và xác minh bằng cách đọc lại file trước khi báo hoàn tất.
10. Nếu người dùng yêu cầu Gemini review hoặc thực hiện một vòng agent, Codex phải chuyển vòng đó sang quy trình Gemini, không âm thầm tự review thay. Nếu Gemini không khả dụng, báo điểm chặn thay vì trình bày kết quả do Codex viết như đầu ra của Gemini.
11. Nếu nội dung đọc hoặc ghi có thể vượt giới hạn PowerShell, Git Bash, terminal hoặc context model/tool, chia công việc thành chunk nhỏ. Đọc file lớn bằng tìm kiếm mục tiêu hoặc khoảng dòng; ghi đầu ra lớn theo batch đã xác minh hoặc theo phần file thay vì một command/prompt khổng lồ, và đọc lại các chunk đã chạm sau mỗi batch.
12. Khi tạo PRD, spec hoặc review HTML, thêm trường metadata `Trạng thái người đọc` mặc định `Chưa đọc`. Người dùng sẽ tự chỉnh trường này sau khi đọc/xử lý, ví dụ `Đã đọc`, `OK`, `Loại bỏ`, `Đã thực thi` hoặc trạng thái phù hợp với loại tài liệu.

## Kế Hoạch Mở Rộng

- Dùng `anfis-guide` khi cần hiểu hoặc mở rộng thiết lập.
- Dùng `anfis-project-agents` khi cần điều phối PRD, spec, task, thực thi, debug hoặc review.
- Chỉ thêm kỹ năng vai trò mới khi quy trình đã ổn định và có thể tái sử dụng.

# LLM Local — DeepSeek-V4-Pro-Qwen3.5-9B (UNCENSORED-heretic)

Bộ triển khai LLM chạy cục bộ hoàn chỉnh bằng Docker Compose: **llama.cpp server** phục vụ suy luận trên GPU (API tương thích OpenAI) + **Open WebUI** làm giao diện chat có **quản lý conversation theo project**.

| | |
|---|---|
| Model | `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` (lấy từ `model-name.txt`) |
| Thể loại | Reasoning model, multimodal (nhận ảnh), ~9.65B tham số, Apache-2.0 |
| Phục vụ | llama.cpp server — bản GGUF **Q4_K_M** (5.24 GiB) của chính model đó |
| Hiệu năng đo được | ~45 tok/s trên RTX 4060 8GB, chiếm ~6.5/8 GB VRAM ở context 32768 |

## Mục lục

1. [Tổng quan hệ thống](#1-tổng-quan-hệ-thống)
2. [Model](#2-model)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Hướng dẫn triển khai](#4-hướng-dẫn-triển-khai)
5. [Sử dụng](#5-sử-dụng)
6. [Cấu hình](#6-cấu-hình)
7. [Vận hành & bảo trì](#7-vận-hành--bảo-trì)
8. [Xử lý sự cố](#8-xử-lý-sự-cố)
9. [Cấu trúc thư mục](#9-cấu-trúc-thư-mục)

---

## 1. Tổng quan hệ thống

```
 Windows / trình duyệt / ứng dụng
        │
        │  http://localhost:3001 (WebUI)      http://localhost:8000/v1 (API)
        ▼                                          ▼
┌─────────────────┐   mạng nội bộ compose   ┌──────────────────┐
│   open-webui    │ ──────────────────────► │  llama-server    │
│  (chat + UI)    │  http://llama-server:   │  (llama.cpp,     │
│  port host 3001 │       8000/v1           │   CUDA, GPU)     │
└────────┬────────┘                         └────────┬─────────┘
         │                                           │
   volume: openwebui-data                volume: model-cache
   (tài khoản, projects, chat)           (GGUF ~5.9 GiB đã tải)
```

| Thành phần | Container | Cổng host | Vai trò |
|---|---|---|---|
| `llama-server` | `llm-local-server` | **8000** | Suy luận GPU, API tương thích OpenAI, WebUI tối giản, `/metrics` Prometheus |
| `open-webui` | `llm-local-webui` | **3001** | Giao diện chat đầy đủ, tài khoản người dùng, **phân loại chat theo project** |

`open-webui` chỉ khởi động sau khi `llama-server` chuyển trạng thái **healthy** (`depends_on: service_healthy`).

**Endpoint:**

| Địa chỉ | Dùng để |
|---|---|
| `http://localhost:3001` | Open WebUI — chat, quản lý project/conversation |
| `http://localhost:8000` | WebUI tối giản của llama-server (test nhanh) |
| `http://localhost:8000/v1` | API tương thích OpenAI (chat completions, models…) |
| `http://localhost:8000/metrics` | Metrics Prometheus |

## 2. Model

| Thuộc tính | Giá trị |
|---|---|
| Tên (theo `model-name.txt`) | `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` |
| Kiến trúc | Qwen3.5 hybrid — 24 lớp linear-attention + 8 lớp full-attention, vocab 248K, context tối đa 262K tokens |
| Khả năng | Reasoning (trả `reasoning_content` + `content`), hiểu ảnh (image-text-to-text) |
| Nguồn gốc | Bản "abliterated" (gỡ bỏ từ chối) của `Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B`, quant lại bởi `mradermacher` |
| License | Apache-2.0 |

**Vì sao chạy bản GGUF Q4_K_M thay vì repo gốc:** bản gốc BF16 nặng ~19GB, không chứa vừa GPU 8GB (RAM máy 15GB cũng không chạy CPU được). Bản GGUF Q4_K_M (5.24 GiB) + mmproj Q8_0 (0.58 GiB, phần vision) là điểm cân bằng tốt nhất cho 8GB VRAM — đã kiểm chứng: trả lời đúng, nhận diện ảnh đúng, ~45 tok/s.

Các bản quant khả dụng trong repo GGUF (kèm mmproj Q8_0 +0.58 GiB mọi bản):

| Quant | Kích thước | Phù hợp |
|---|---|---|
| `Q2_K` | 3.56 GiB | Chất lượng giảm nhiều — chỉ khi cực thiếu VRAM |
| `IQ4_XS` | 4.87 GiB | Tiết kiệm hơn Q4_K_M, chất lượng giảm nhẹ |
| **`Q4_K_M`** | **5.24 GiB** | **Mặc định — cân bằng tốt nhất cho 8GB** |
| `Q5_K_M` | 6.02 GiB | Chất lượng cao hơn, 8GB VRAM rất chật |
| `Q6_K` / `Q8_0` | 6.85 / 8.87 GiB | Cần GPU ≥ 12GB / ≥ 13GB |

## 3. Yêu cầu hệ thống

**Phần cứng** (định mức cho Q4_K_M):

| | Tối thiểu | Máy đã triển khai |
|---|---|---|
| GPU VRAM | 8 GB | RTX 4060 8GB |
| RAM | 8 GB | 15 GB |
| Ổ đĩa trống | ~15 GB (model + image) | Docker root còn ~170 GB |
| CPU | 4 nhân | 12 nhân |

**Phần mềm:**

- Docker Engine + Docker Compose v2
- NVIDIA driver (máy này: 560.94 / CUDA 12.6)
- **`nvidia-container-toolkit`** để Docker thấy GPU — xem Bước 2 nếu chưa có
- Hai cổng trống trên **cả WSL lẫn Windows**: `8000` và `3001`

> Máy này chọn 8000 vì 8080 bị `csy-media-engine` chiếm, 11434 bị `csy-ollama` chiếm; chọn 3001 vì **3000 trên Windows bị một app Node.js chiếm** (gọi `localhost:3000` sẽ vào app đó, không phải stack này).

## 4. Hướng dẫn triển khai

### Bước 1 — Kiểm tra GPU nhìn thấy được trong Docker

```bash
docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

Hiện bảng NVIDIA GPU = đạt. Nếu báo `failed to discover GPU vendor from CDI` → làm Bước 2, ngược lại sang Bước 3.

### Bước 2 — Cài nvidia-container-toolkit (một lần, cần sudo)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker   # mọi container đang chạy sẽ khởi động lại
```

### Bước 3 — Rà soát cấu hình `.env`

Các giá trị mặc định đã đúng cho máy 8GB VRAM; chỉ cần đổi nếu muốn (xem [Mục 6](#6-cấu-hình)).

### Bước 4 — Khởi động stack

```bash
docker compose up -d
docker compose ps          # đợi cả 2 service sang (healthy)
```

Lần đầu chạy mất ~10 phút: `llama-server` tải ~5.9 GiB model từ Hugging Face vào volume `model-cache` (**log giai đoạn này gần như im lặng — bình thường**, theo dõi bằng `docker compose logs -f` hoặc xem dung lượng cache tăng: `docker exec llm-local-server du -sh /root/.cache`). Các lần sau chỉ mất ~15–30 giây load vào GPU.

### Bước 5 — Kiểm tra triển khai

```bash
# API sống chưa (/health: 503 = đang load, 200 = sẵn sàng)
curl http://localhost:8000/health

# Chat thử
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "Trả lời ngắn: thủ đô Việt Nam?"}]
}'
# kỳ vọng: content = "Hà Nội", kèm reasoning_content (model suy luận trước khi trả lời)

# WebUI
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3001/   # kỳ vọng 200
```

### Bước 6 — Tài khoản Open WebUI

Lần đầu vào `http://localhost:3001` sẽ yêu cầu **Sign up** — **tài khoản đầu tiên tự động là admin**.

> **Máy này đã thiết lập:** `admin@llm.local` (mật khẩu được cấp trong phiên triển khai — nên đổi trong Settings → Account sau khi đăng nhập lần đầu). Chỉ dùng thông tin này khi volume `openwebui-data` còn nguyên; người dùng khác tự Sign up.

## 5. Sử dụng

### 5.1. Chat & quản lý conversation theo project (Open WebUI)

Mở `http://localhost:3001`:

- **Tạo project**: sidebar → mục **Folders** → nút **+** → đặt tên.
- **Tạo conversation trong project**: chọn folder rồi **New Chat**, hoặc tạo chat thường rồi thả vào folder.
- **Di chuyển conversation**: kéo-thả chat vào folder (từ "ngoài" vào project, hoặc project này sang project khác), hoặc right-click chat → **Move to** → chọn project.
- Conversation chưa thuộc project nào nằm ở mục **All chats**; right-click folder để đổi tên/xóa.
- Gửi ảnh trực tiếp trong chat (multimodal).

Dữ liệu tài khoản/projects/conversations nằm trong volume `openwebui-data` — không mất khi restart hay `docker compose down`.

Muốn bỏ bước đăng nhập: bỏ dấu `#` trước `WEBUI_AUTH: "false"` trong `docker-compose.yml` (không khuyến nghị nếu máy truy cập được từ LAN).

### 5.2. API tương thích OpenAI

Python (thư viện `openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
r = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{"role": "user", "content": "Giải phương trình x² - 5x + 6 = 0"}],
)
print(r.choices[0].message.reasoning_content)  # phần suy luận
print(r.choices[0].message.content)            # câu trả lời
```

Gửi ảnh (multimodal):

```python
r = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Mô tả ảnh này"},
            {"type": "image_url", "image_url": {"url": "https://example.com/anh.jpg"}},
        ],
    }],
)
```

### 5.3. Giám sát

`http://localhost:8000/metrics` (Prometheus) — số request, token/s, slot, RAM/VRAM dùng.

## 6. Cấu hình

Toàn bộ nằm trong **`.env`** (không cần sửa `docker-compose.yml`):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MODEL_ID` | `xtools-at/…-heretic` | Model gốc theo `model-name.txt` — chỉ để truy vết |
| `GGUF_REPO` | `mradermacher/…-GGUF` | Repo GGUF dùng để chạy |
| `QUANT` | `Q4_K_M` | Bản quant (xem bảng [Mục 2](#2-model)) |
| `SERVED_MODEL_NAME` | `deepseek-v4-pro` | Tên model trong API (`"model": "…"`) |
| `HOST_PORT` | `8000` | Cổng API llama-server trên host |
| `WEBUI_PORT` | `3001` | Cổng Open WebUI trên host |
| `CTX_SIZE` | `32768` | Độ dài context (tokens) — KV cache ~32KB/token nên 8GB VRAM vẫn đủ. Cần 64K+ → giảm `QUANT=IQ4_XS` hoặc tắt vision; thiếu VRAM → giảm 4096 |
| `FIT_TARGET` | `512` | MiB VRAM giữ dự phòng khi llama-server tự cân chỉnh ctx/số layer trên GPU |
| `LLAMA_IMAGE` | `…:server-cuda` | Image suy luận (đổi `…:server` + bỏ khối GPU khi chạy CPU) |
| `OPENWEBUI_IMAGE` | `…:main` | Image Open WebUI |

Các tùy chọn khác — bỏ dấu `#` trong `docker-compose.yml`:

- **Tắt vision** (nhường ~0.58 GiB VRAM): `LLAMA_ARG_MMPROJ_AUTO: 0`
- **Bảo vệ API bằng key**: `LLAMA_API_KEY: ${API_KEY:-}` + đặt `API_KEY` trong `.env`; client gọi kèm `Authorization: Bearer <key>`
- **HF_TOKEN**: chỉ cần nếu đổi sang model private

Đổi cấu hình xong chạy `docker compose up -d` để áp dụng.

## 7. Vận hành & bảo trì

```bash
docker compose ps                          # trạng thái + health
docker compose logs -f [llama-server]      # theo dõi log
docker compose restart                     # restart cả stack
docker compose down                        # dừng (giữ nguyên volume + dữ liệu)
docker compose pull && docker compose up -d   # cập nhật image
```

**Sao lưu dữ liệu WebUI** (tài khoản, projects, conversations):

```bash
docker run --rm -v llm-local_openwebui-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/openwebui-backup.tar.gz -C /data .
```

**Đổi model khác:** sửa `GGUF_REPO` + `QUANT` + `SERVED_MODEL_NAME` trong `.env` → `docker compose up -d`. Model cũ vẫn nằm trong volume `model-cache` (tiện quay lại; xóa hẳn bằng `docker volume rm llm-local_model-cache`).

**Chạy tạm khi không có GPU:** `LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server` trong `.env`, **xóa khối `devices`** của service `llama-server` trong `docker-compose.yml`, rồi `up -d`. Tốc độ CPU ~vài token/giây — chỉ dùng tạm.

**Dung lượng:** volume `model-cache` ~5.9 GiB/bản quant; image llama.cpp CUDA ~vài GB; Open WebUI ~4 GB.

## 8. Xử lý sự cố

**1. `failed to discover GPU vendor from CDI` khi start** — thiếu `nvidia-container-toolkit`, cài theo [Bước 2](#bước-2--cài-nvidia-container-toolkit-một-lần-cần-sudo) (đã gặp trên máy này trước khi triển khai).

**2. Lỗi `request (N tokens) exceeds the available context size (8192 tokens)`** — prompt + lịch sử chat vượt `CTX_SIZE`. Đã nâng mặc định lên 32768 (kiểm chứng thực tế: request 15K tokens chạy tốt, VRAM 6.5/8GB). Cần lớn hơn nữa: tăng `CTX_SIZE` kèm `QUANT=IQ4_XS` hoặc tắt vision; hoặc cắt bớt nội dung gửi. Kiểm tra ctx đang chạy: `curl -s localhost:8000/props | grep -o '"n_ctx":[0-9]*'`.

**3. Hết VRAM / container restart liên tục** — lần lượt: `CTX_SIZE=4096` → `QUANT=IQ4_XS` → tắt vision (`LLAMA_ARG_MMPROJ_AUTO: 0`). llama-server có cơ chế tự "fit" nên thường chỉ chậm đi chứ không chết.

**4. Cổng xung đột** — kiểm tra WSL: `ss -tln | grep <cổng>`; kiểm tra Windows: `netstat -ano | findstr :<cổng>`. Đổi cổng trong `.env` rồi `docker compose up -d`. *Trường hợp thật trên máy này:* `localhost:3000` trả JSON `{"error":"Truy cập từ địa chỉ IP của bạn bị chặn bởi quản trị viên."}` — do một app Node.js trên Windows chiếm cổng 3000, không phải stack này.

**5. Windows không mở được `localhost:3001`/`8000` dù WSL vẫn chạy** — WSL2 hỏng tính năng forward localhost (đã gặp: `127.0.0.1:8000` từ Windows bị refuse). Sửa: chạy **`wsl --shutdown`** trong PowerShell rồi mở lại terminal WSL — systemd tự kéo Docker + container lên lại (downtime ~1 phút cho mọi stack trong WSL, IP WSL sẽ đổi). Cách tạm: lấy IP bằng `hostname -I` trong WSL rồi truy cập `http://<IP-WSL>:3001`.

**6. Lần đầu chạy "đơ" ở 2 dòng log** — đang tải ~5.9 GiB model, log im lặng là bình thường. Xem tiến độ: `docker exec llm-local-server du -sh /root/.cache` (tới ~5.9G là xong).

**7. Muốn chạy lại từ đầu sạch sẽ** — `docker compose down -v` (**xóa cả dữ liệu**: model cache + tài khoản/projects/chat).

## 9. Cấu trúc thư mục

```
llm-local/
├── model-name.txt        # model gốc cần triển khai (đầu bài)
├── docker-compose.yml    # 2 service + 2 volume, GPU reservation, healthcheck
├── .env                  # toàn bộ cấu hình (model, quant, cổng, context…)
└── README.md             # tài liệu này
```

Volume Docker (ngoài thư mục): `llm-local_model-cache`, `llm-local_openwebui-data`.

---

*Ghi chú: GGUF là bản quant lại của model gốc do bên thứ ba (`mradermacher`, ~378K lượt tải) phát hành — cùng weights, nén 4-bit; model gốc là bản "abliterated" (đã gỡ bỏ cơ chế từ chối) — người vận hành tự chịu trách nhiệm cách sử dụng. Stack chạy độc lập, song song với Ollama sẵn có trên máy.*

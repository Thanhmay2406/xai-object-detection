# Runbook vận hành GPU Kaggle

Tài liệu này mô tả cách dùng Kaggle làm máy thực thi GPU cho repository
`xai-object-detection` từ terminal Arch Linux. Kaggle runner chỉ clone mã nguồn,
cài dependency cần thiết và gọi entry point có sẵn; nó không thay thế logic
nghiên cứu trong `src/xai_pruning/`.

Tài liệu được kiểm tra với Kaggle CLI 2.2.2 và NVIDIA Tesla T4 ngày
2026-09-05.

## 1. Luồng vận hành

```text
repository local
  -> commit/push mã cần chạy lên origin/main
  -> scripts/kaggle_run.sh
  -> Kaggle kernel clone repository công khai
  -> GPU Tesla T4
  -> artifact trong /kaggle/working/xai_pruning_outputs
  -> scripts/kaggle_pull.sh
  -> results/kaggle/<run-id>
```

Các file điều khiển chính:

- `configs/kaggle.yaml`: mode, entry point và thư mục artifact.
- `kaggle/kernel-metadata.json`: loại kernel, internet và accelerator.
- `kaggle/runner.py`: bootstrap mỏng chạy trên Kaggle.
- `scripts/kaggle_run.sh`: kiểm tra Git rồi push kernel.
- `scripts/kaggle_status.sh`: xem trạng thái phiên bản mới nhất.
- `scripts/kaggle_pull.sh`: tải output, không dùng chế độ ghi đè.

## 2. Chuẩn bị terminal

Chạy từ thư mục gốc repository:

```bash
cd /home/thanhmay/workspace/xai-object-detection
export PATH="$PWD/.venv/bin:$PATH"
export KAGGLE_KERNEL_ID="thanhmay2406/xai-pruning-runner"
```

Kiểm tra CLI và đăng nhập:

```bash
kaggle --version
kaggle kernels list --mine
```

Nếu phiên đăng nhập không còn hợp lệ:

```bash
kaggle auth login
```

Không chạy `kaggle auth print-access-token` trong log chia sẻ. Không `cat`, copy
hoặc commit `~/.kaggle/kaggle.json`, access token, OAuth credential hay file
`.env`.

## 3. Kiểm tra trước khi dùng GPU

### 3.1. Kiểm tra Git

Kaggle clone GitHub, không đọc trực tiếp worktree local. Mọi mã cần chạy phải có
trên remote trước khi launch:

```bash
git status --short
git diff --check
git rev-parse HEAD
git rev-parse '@{upstream}'
```

Chỉ commit các file đã review. Không đưa dataset, checkpoint, credential hoặc
kết quả local vào commit. `scripts/kaggle_run.sh` sẽ từ chối launch nếu các file
deployment có thay đổi chưa publish hoặc `HEAD` khác upstream.

Không dùng `KAGGLE_SKIP_GIT_CHECK=1` trừ khi đã xác minh chắc chắn remote chứa
đúng toàn bộ mã cần thiết.

### 3.2. Kiểm tra mode

Smoke test an toàn hiện tại phải giữ:

```yaml
execution:
  mode: smoke_test
```

Kiểm tra nhanh:

```bash
rg -n '^\s*mode:\s*smoke_test$' configs/kaggle.yaml
```

Trong mode này runner chỉ thực hiện:

- nhận diện Python, PyTorch, CUDA và GPU;
- cấp phát tensor CUDA 16 x 16 và nhân ma trận nhỏ;
- import `xai_pruning`;
- import API reconstruction và evaluation;
- ghi `experiment.json` và `logs/runner.log`.

Nó không train, fine-tune, prune, chạy XAI, đọc dataset hoặc dựng model từ
checkpoint.

### 3.3. Kiểm tra T4

`kaggle/kernel-metadata.json` phải có:

```json
{
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4"
}
```

Luôn truyền `NvidiaTeslaT4` khi muốn override rõ ràng. Không chuyển sang P100 với
Kaggle image PyTorch mặc định hiện tại: tài liệu Kaggle cảnh báo build CUDA 12.8
không có kernel Pascal `sm_60`, nên `torch.cuda.is_available()` có thể đúng nhưng
phép toán CUDA đầu tiên vẫn lỗi.

## 4. Launch smoke test

```bash
./scripts/kaggle_run.sh NvidiaTeslaT4
```

`kaggle kernels push` vừa upload vừa tạo một phiên bản kernel mới và bắt đầu
chạy ngay. Ghi lại dòng trả về, ví dụ:

```text
Kernel version 2 successfully pushed.
```

Mỗi lần push là một version mới và có thể tiêu thụ quota. Không retry liên tục
nếu chưa đọc log của version lỗi.

## 5. Theo dõi trạng thái và log

Trạng thái phiên bản mới nhất:

```bash
./scripts/kaggle_status.sh
```

Theo dõi log đến khi kết thúc:

```bash
kaggle kernels logs "$KAGGLE_KERNEL_ID" --follow --interval 5
```

Để khóa đúng một version khi điều tra:

```bash
kaggle kernels logs "${KAGGLE_KERNEL_ID}/2" --follow --interval 5
```

Các dòng bắt buộc đối với smoke test thành công:

```text
CUDA available: True
GPU: Tesla T4
CUDA tensor test: PASS
xai_pruning import: PASS
evaluation API import: PASS
reconstruction API import: PASS
GPU smoke test: PASS
Artifacts: /kaggle/working/xai_pruning_outputs
```

Chỉ kết luận thành công sau khi status là `KernelWorkerStatus.COMPLETE` và output
đã được tải về, đọc và kiểm tra.

## 6. Tải và kiểm tra artifact

Mỗi run nên dùng một thư mục mới để bảo toàn lịch sử:

```bash
./scripts/kaggle_pull.sh results/kaggle/smoke-002
```

Script không dùng `--force`, vì vậy không tự ghi đè kết quả cũ. Nó dùng
`--page-size 200`; tùy chọn này cần thiết vì Kaggle CLI mặc định phân trang danh
sách output và có thể bỏ sót artifact ở trang sau.

Các artifact cốt lõi:

```text
results/kaggle/<run-id>/xai_pruning_outputs/
├── experiment.json
└── logs/
    └── runner.log
```

Do repository hiện được clone trong `/kaggle/working`, output tải về cũng có thể
chứa thư mục `xai-object-detection/`. Đây là bản checkout thực thi, không phải
kết quả khoa học.

Kiểm tra JSON bằng Python:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("results/kaggle/smoke-002/xai_pruning_outputs/experiment.json")
record = json.loads(path.read_text(encoding="utf-8"))
smoke = record["smoke_test"]

assert record["status"] == "completed"
assert record["mode"] == "smoke_test"
assert record["inside_kaggle"] is True
assert smoke["cuda_available"] is True
assert smoke["gpu_name"] == "Tesla T4"
assert smoke["cuda_tensor_test"] == "passed"
assert smoke["xai_pruning_import"] == "passed"
assert smoke["evaluation_api_import"] == "passed"
assert smoke["reconstruction_api_import"] == "passed"
print("KAGGLE_GPU_SMOKE_TEST=PASS")
PY
```

## 7. Quota và sử dụng GPU có kiểm soát

Lệnh chuẩn của CLI là:

```bash
kaggle quota
```

Ở Kaggle CLI 2.2.2 trên máy này, cả output bảng và CSV hiện lỗi:

```text
not enough values to unpack (expected 2, got 1)
```

Đây là lỗi hiển thị quota của CLI, không phải bằng chứng quota bằng không. Khi
gặp lỗi này, kiểm tra quota trong giao diện tài khoản Kaggle trước khi launch
experiment dài.

Nguyên tắc sử dụng:

- smoke trước, mini pipeline sau, full experiment cuối cùng;
- đọc log và tìm root cause trước mỗi retry;
- không đổi hoặc reinstall PyTorch/CUDA chỉ để xử lý lỗi path/import;
- không dùng GPU cho thao tác có thể kiểm tra bằng syntax/unit test local;
- đặt mọi artifact cần giữ dưới `/kaggle/working`;
- dùng run ID riêng cho từng seed, phương pháp pruning và config.

Kaggle CLI 2.2.2 không cung cấp lệnh `kernels stop/cancel` trong nhóm lệnh
`kernels`. Nếu cần dừng một phiên đang chạy, dùng điều khiển phiên trong giao
diện Kaggle. Không dùng `kaggle kernels delete`: đó là thao tác xóa kernel, không
phải dừng run.

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `kaggle: command not found` | `.venv/bin` chưa có trong `PATH` | `export PATH="$PWD/.venv/bin:$PATH"` |
| API báo chưa xác thực | phiên đăng nhập hết hạn | chạy `kaggle auth login`; không gửi token qua chat |
| script chặn vì unpublished changes | mã Kaggle sẽ clone chưa có trên remote | review, commit và push đúng file cần thiết |
| `ModuleNotFoundError: xai_pruning` | Python process chưa thấy src-layout sau editable install | dùng runner từ commit `7838c2d` trở lên; log phải có `Project source path: .../src` |
| `CUDA available: False` | kernel không được cấp GPU hoặc metadata sai | kiểm tra T4 metadata rồi tạo version mới |
| CUDA báo không có kernel image trên P100 | PyTorch mặc định không hỗ trợ `sm_60` | dùng T4; không blindly reinstall core GPU packages |
| clone GitHub thất bại | internet bị tắt hoặc URL/branch sai | kiểm tra `enable_internet`, repo URL và branch `main` |
| pull thiếu `experiment.json` | output bị phân trang | dùng script hiện tại hoặc `--page-size 200` |
| output local đã tồn tại | script cố bảo vệ kết quả cũ | chọn thư mục run mới; không dùng `--force` mặc định |

## 9. Chuyển sang experiment thật trong tương lai

Không đổi `execution.mode` trong một smoke-test task. Khi bắt đầu experiment
thật, thực hiện ở một thay đổi riêng và review tối thiểu:

1. entry point có tồn tại và dùng chung implementation trong package;
2. dataset/checkpoint được attach qua Kaggle input, không commit vào Git;
3. config ghi rõ split, seed, checkpoint và output directory;
4. entry point chỉ ghi file cần giữ dưới `/kaggle/working`;
5. chạy mini end-to-end với vài mẫu trước;
6. chỉ sau khi mini test đạt mới launch full experiment;
7. tải artifact về thư mục run riêng và ghi lại Git commit + kernel version.

Smoke test chỉ chứng minh môi trường GPU và import hoạt động. Nó không chứng minh
mAP, độ đúng của pruning, tốc độ, mức dùng bộ nhớ, hội tụ fine-tuning hoặc giá trị
khoa học của experiment.

## 10. Trạng thái chuẩn đã xác nhận

Run tham chiếu đầu tiên:

```text
Kernel: thanhmay2406/xai-pruning-runner
Version: 2
Accelerator: Tesla T4
Repository commit: 7838c2de960fc8dd5107770fe859cef453df1124
Python: 3.12.13
PyTorch: 2.10.0+cu128
CUDA tensor checksum: 4096.0
Status: completed
```

Artifact tham chiếu nằm tại:

```text
results/kaggle/smoke-001/xai_pruning_outputs/experiment.json
results/kaggle/smoke-001/xai_pruning_outputs/logs/runner.log
```

## 11. Tài liệu chính thức

- Kaggle CLI kernel commands: <https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md>
- Kaggle kernel metadata: <https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md>
- Kaggle authentication: <https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md>

# RITM Interactive Segmentation

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7 hoặc 3.8
- Windows 10/11
- Visual C++ Build Tools (cho Windows)

### Cài đặt thủ công (Khuyến nghị)

1. **Tạo môi trường Python mới**
```bash
python -m venv venv
venv\Scripts\activate
```

2. **Cài đặt PyTorch CPU version**
```bash
pip install torch==1.9.0 torchvision==0.10.0 --index-url https://download.pytorch.org/whl/cpu
```

3. **Cài đặt các thư viện cơ bản**
```bash
pip install numpy==1.21.0
pip install Pillow==8.3.1
pip install opencv-python==4.5.3.56
pip install scipy
pip install Cython
pip install scikit-image
pip install matplotlib
pip install imgaug>=0.4
pip install albumentations>0.5
pip install graphviz
pip install tqdm
pip install pyyaml
pip install easydict
pip install tensorboard
pip install future
pip install cffi
pip install ninja
```

4. **Cài đặt FastAPI và các dependencies**
```bash
pip install fastapi
pip install uvicorn
pip install python-multipart
```

5. **Cài đặt các thư viện phụ trợ**
```bash
pip install flask==2.0.1
```

### Cài đặt nhanh (Sử dụng requirements.txt)
```bash
pip install -r requirements.txt
```

### Kiểm tra cài đặt
Để kiểm tra cài đặt thành công, chạy Python và thử:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # Sẽ in False nếu dùng CPU
```

## Chạy ứng dụng

1. **Khởi động API server**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

2. **Truy cập API documentation**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Các API Endpoints

- `/init-session`: Khởi tạo session với model và cấu hình
- `/upload-image`: Upload ảnh để xử lý
- `/click`: Xử lý click của người dùng để tạo mask
- `/reset-clicks`: Reset lại các click
- `/undo-click`: Undo click cuối cùng
- `/finish-object`: Hoàn thành việc segment một object
- `/get-mask`: Lấy mask hiện tại
- `/get-pure-mask`: Lấy mask thuần túy
- `/get-segmented-image`: Lấy ảnh đã được segment

## Cấu hình mặc định

- Model: `models/coco_lvis_h18_baseline.pth`
- Device: CPU
- Giới hạn kích thước ảnh dài nhất: 800px
- File cấu hình: `config.yml`

## Xử lý lỗi thường gặp

1. **Lỗi khi cài đặt Cython**
- Cài đặt Visual C++ Build Tools
- Tải từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. **Lỗi khi cài đặt ninja**
- Có thể bỏ qua vì không bắt buộc
- Hoặc cài đặt thủ công từ: https://github.com/ninja-build/ninja/releases

3. **Lỗi quyền truy cập**
- Chạy Command Prompt với quyền Administrator
- Hoặc sử dụng `pip install --user` cho các gói

4. **Lỗi khi chạy server**
- Đảm bảo port 8000 không bị sử dụng
- Kiểm tra file model tồn tại trong thư mục `models/`

from fastapi import FastAPI     # framework tạo API
from pydantic import BaseModel  # Kiểm tra dữ liệu đầu vào
import joblib                   # Thư viện để load model đã được huấn luyện sẵn
import pandas as pd             # Thư viện xử lý dữ liệu dạng bảng
import numpy as np              # Thư viện xử lý dữ liệu dạng mảng
import csv                      # Thư viện để ghi log vào file CSV
import os                       # Thư viện để kiểm tra sự tồn tại của file
from datetime import datetime   # Thư viện để lấy thời gian hiện tại

# 1. Khởi tạo app
app = FastAPI()

# 2. Load model từ Dev 1 tạo ra
model = joblib.load("models/model.pkl")

# 3. Định nghĩa dữ liệu đầu vào
class InputData(BaseModel):
    thu_nhap: float
    so_tien_vay: float
    thoi_han_vay: int
    diem_tin_dung: int

# 4. Endpoint dự đoán
@app.post("/predict")
def predict(data: InputData):

    # Tính lãi suất năm và lãi suất tháng dựa trên điểm tín dụng của người dùng
    lai_suat_nam = 0.08 + (1 - (data.diem_tin_dung / 850)) * 0.15 
    lai_suat_thang = lai_suat_nam / 12

    # Tính số tiền trả hàng tháng (EMI)
    tks = (1 + lai_suat_thang) ** data.thoi_han_vay
    tra_hang_thang = data.so_tien_vay * lai_suat_thang * tks / (tks - 1)

    # Tạo bảng mới, gom đủ 5 cột
    input_df = pd.DataFrame([{
        'thu_nhap': int(data.thu_nhap),
        'so_tien_vay': int(data.so_tien_vay),
        'thoi_han_vay': int(data.thoi_han_vay),
        'diem_tin_dung': int(data.diem_tin_dung),
        'tra_hang_thang': int(tra_hang_thang)  # 🌟 Cột hệ thống vừa tự tính ở Phần 2
    }])

    # 5. Tiến hành dự đoán nợ xấu (0: Tốt, 1: Nợ xấu)
    ket_qua = model.predict(input_df)[0]

    # 6. Ghi log lịch sử dự đoán
    log_path = "logs/inference_logs.csv"
    os.makedirs("logs", exist_ok=True) # Đảm bảo thư mục logs tồn tại
    file_exists = os.path.exists(log_path)
    
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["thoi_gian", "thu_nhap", "so_tien_vay", "thoi_han_vay", "diem_tin_dung", "tra_hang_thang", "ket_qua"])
        writer.writerow([
            datetime.now(), 
            data.thu_nhap, 
            data.so_tien_vay, 
            data.thoi_han_vay, 
            data.diem_tin_dung, 
            int(tra_hang_thang), 
            ket_qua
        ])

    # 7. Trả về kết quả
    return {"ket_qua_du_doan": int(ket_qua)}
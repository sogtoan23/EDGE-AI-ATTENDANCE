import os
import json
from cryptography.fernet import Fernet
from database import SessionLocal, Student

# 1. Đọc khóa mã hóa từ biến môi trường hoặc file secret.key
def load_fernet_key() -> bytes:
    env_key = os.getenv("FERNET_KEY")
    if env_key:
        return env_key.encode("utf-8")
    
    if os.path.exists("secret.key"):
        with open("secret.key", "rb") as f:
            return f.read().strip()
    
    raise FileNotFoundError("Không tìm thấy file secret.key hoặc biến môi trường FERNET_KEY!")

# 2. Khởi tạo bộ giải mã
key = load_fernet_key()
cipher_suite = Fernet(key)

# 3. Kết nối CSDL và đọc toàn bộ sinh viên
db = SessionLocal()
try:
    students = db.query(Student).all()
    print(f"\n{'='*75}")
    print(f"       TỔNG SỐ SINH VIÊN TRONG CSDL: {len(students)}")
    print(f"{'='*75}\n")

    for idx, s in enumerate(students, 1):
        print(f"[{idx}] Mã SV: {s.student_id} | Tên: {s.full_name} | Lớp: {getattr(s, 'class_name', 'N/A')}")
        
        # Kiểm tra xem trường vector có tồn tại không
        if not s.embedding_json:
            print("    ❌ Chưa có dữ liệu mã hóa trong DB (Trường rỗng)\n")
            continue

        raw_cipher = str(s.embedding_json).strip()
        print(f"    🔒 Chuỗi mã hóa trong DB (rút gọn): {raw_cipher[:35]}...{raw_cipher[-15:]}")

        try:
            # Giải mã bằng khóa Fernet
            decrypted_bytes = cipher_suite.decrypt(raw_cipher.encode("utf-8"))
            decrypted_text = decrypted_bytes.decode("utf-8")
            
            # Chuyển chuỗi JSON thành mảng List các số thực
            vector_512 = json.loads(decrypted_text)

            print(f"    ✅ GIẢI MÃ THÀNH CÔNG:")
            print(f"       • Số chiều (Dimension): {len(vector_512)} chiều")
            print(f"       • 5 giá trị đầu       : {[round(x, 4) for x in vector_512[:5]]}")
            print(f"       • 5 giá trị cuối      : {[round(x, 4) for x in vector_512[-5:]]}")
            
            # Nếu muốn in ra toàn bộ vector 512 số thực:
            # print(f"       • Chi tiết vector: {vector_512}")

        except Exception as e:
            print(f"    ❌ LỖI GIẢI MÃ: {e}")
            print("       (Nguyên nhân: Bản ghi này được tạo bằng khóa tạm cũ trước khi tạo secret.key cố định)")

        print("-" * 75)

finally:
    db.close()
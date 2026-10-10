import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from cryptography.fernet import Fernet
import bcrypt
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# -------------------------------------------------------------
# 1. CƠ CHẾ TỰ ĐỘNG SINH KHÓA & LƯU CỐ ĐỊNH VÀO FILE HOẶC BIẾN MÔI TRƯỜNG
# -------------------------------------------------------------
KEY_FILE_PATH = "secret.key"

def get_or_create_fernet_key() -> bytes:
    """
    1. Ưu tiên biến môi trường FERNET_KEY (tiện cấu hình trên Render).
    2. Nếu không có biến môi trường, đọc từ file secret.key.
    3. Nếu chưa có file secret.key, tự động tạo mới và lưu vào file.
    """
    env_key = os.getenv("FERNET_KEY")
    if env_key:
        return env_key.encode("utf-8")

    if os.path.exists(KEY_FILE_PATH):
        with open(KEY_FILE_PATH, "rb") as f:
            key = f.read().strip()
            if key:
                return key

    # Tự động sinh khóa Fernet chuẩn 32-byte URL-safe base64 một lần duy nhất
    new_key = Fernet.generate_key()
    with open(KEY_FILE_PATH, "wb") as f:
        f.write(new_key)
    print(f"[SECURITY] Đã tạo mới và lưu khóa mã hóa vĩnh viễn vào file: {KEY_FILE_PATH}")
    return new_key

# Khởi tạo đối tượng mã hóa giải mã Fernet cố định
FERNET_KEY = get_or_create_fernet_key()
cipher_suite = Fernet(FERNET_KEY)

def encrypt_vector_string(plain_text: str) -> str:
    """Mã hóa chuỗi JSON vector 512-D thành chuỗi bảo mật."""
    return cipher_suite.encrypt(plain_text.encode("utf-8")).decode("utf-8")

def decrypt_vector_string(cipher_text: str) -> str:
    """Giải mã chuỗi bảo mật trở lại thành chuỗi JSON vector 512-D."""
    return cipher_suite.decrypt(cipher_text.encode("utf-8")).decode("utf-8")


# -------------------------------------------------------------
# 2. BẢO MẬT MẬT KHẨU (DÙNG TRỰC TIẾP BCRYPT THUẦN - BỎ PASSLIB)
# -------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash mật khẩu (giới hạn tối đa 72 bytes theo chuẩn bcrypt)."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác thực mật khẩu với chuỗi hash bcrypt."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode("utf-8"))
    except Exception:
        return False

# Mật khẩu quản trị mặc định: "admin123"
ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    hash_password("admin123")
)


# -------------------------------------------------------------
# 3. JWT ACCESS TOKEN
# -------------------------------------------------------------
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ATTENDANCE_SYSTEM_SECRET_KEY_JWT_2026_XYZ")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 giờ

security_bearer = HTTPBearer(auto_error=False)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập đã hết hạn, vui lòng đăng nhập lại!"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token xác thực không hợp lệ!"
        )


# -------------------------------------------------------------
# 4. DEPENDENCY PHÂN QUYỀN (GIẢNG VIÊN / SINH VIÊN)
# -------------------------------------------------------------
def get_current_admin(auth: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)) -> dict:
    """Xác thực token Giảng viên (Admin)."""
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yêu cầu token xác thực của Giảng viên!"
        )
    payload = decode_token(auth.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền quản trị viên / giảng viên!"
        )
    return payload

def get_current_student(auth: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)) -> dict:
    """Xác thực token Sinh viên."""
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yêu cầu đăng nhập tài khoản sinh viên!"
        )
    payload = decode_token(auth.credentials)
    if payload.get("role") != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Quyền truy cập không hợp lệ!"
        )
    return payload
import os
from datetime import datetime, timedelta
import bcrypt
import jwt
from cryptography.fernet import Fernet
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "ENTERPRISE_SECRET_KEY_SUPER_SECURE_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

# Mật khẩu quản trị mặc định: admin123 (được băm bcrypt)
ADMIN_PASSWORD_HASH = os.getenv(
    "ADMIN_PASSWORD_HASH",
    bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode("utf-8")
)

# Khóa đối xứng Fernet dùng để mã hóa vector 512-D
FERNET_KEY = os.getenv("FERNET_KEY", Fernet.generate_key().decode())
cipher_suite = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

security_bearer = HTTPBearer(auto_error=False)

def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc phiên đăng nhập đã hết hạn!",
            headers={"WWW-Authenticate": "Bearer"}
        )

def get_current_admin(auth: HTTPAuthorizationCredentials = Depends(security_bearer)) -> dict:
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yêu cầu quyền Giảng viên / Quản trị viên!")
    payload = decode_token(auth.credentials)
    if payload.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Từ chối truy cập: Không đủ quyền quản trị!")
    return payload

def get_current_student(auth: HTTPAuthorizationCredentials = Depends(security_bearer)) -> dict:
    if not auth:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Yêu cầu đăng nhập tài khoản sinh viên!")
    payload = decode_token(auth.credentials)
    if payload.get("role") != "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Phiên đăng nhập không thuộc quyền sinh viên!")
    return payload

def encrypt_vector_string(plain_text: str) -> str:
    return cipher_suite.encrypt(plain_text.encode()).decode()

def decrypt_vector_string(cipher_text: str) -> str:
    return cipher_suite.decrypt(cipher_text.encode()).decode()
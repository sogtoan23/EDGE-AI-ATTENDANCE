import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Time, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Mặc định sử dụng SQLite nếu chưa cấu hình biến môi trường DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./face_attendance.db")

IS_SQLITE = DATABASE_URL.startswith("sqlite")

if IS_SQLITE:
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # Nếu dùng PostgreSQL: Hỗ trợ connection pool đa kết nối
    engine = create_engine(
        DATABASE_URL,
        pool_size=20,
        max_overflow=15,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Kích hoạt pgvector extension nếu đang kết nối PostgreSQL
if not IS_SQLITE:
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception as e:
        print(f"[DB Warning] Không thể khởi tạo extension vector: {e}")

# ==============================================================
# BẢNG THIẾT BỊ CAMERA
# ==============================================================
class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), unique=True, index=True, nullable=False)
    device_name = Column(String(100), nullable=True)
    room_name = Column(String(50), index=True, nullable=False)
    device_token = Column(String(100), unique=True, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==============================================================
# BẢNG SINH VIÊN
# ==============================================================
class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    class_name = Column(String(50), index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # Mật khẩu sinh viên (băm bcrypt)
    embedding_json = Column(Text, nullable=False)       # Vector 512-D mã hóa AES
    created_at = Column(DateTime, default=datetime.utcnow)

# ==============================================================
# BẢNG LỊCH HỌC
# ==============================================================
class ClassSchedule(Base):
    __tablename__ = "class_schedules"
    id = Column(Integer, primary_key=True, index=True)
    room_name = Column(String(50), index=True, nullable=False)
    class_name = Column(String(50), index=True, nullable=False, default="")  # Lớp được dạy
    subject_name = Column(String(100), nullable=False)
    teacher_name = Column(String(100), nullable=False)
    day_of_week = Column(Integer, nullable=False)      # 0: Thứ 2, ..., 6: Chủ Nhật
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    early_arrival_minutes = Column(Integer, default=30) # Số phút đến sớm cho phép
    grace_period_minutes = Column(Integer, default=15)  # Số phút đi trễ cho phép

# ==============================================================
# BẢNG ĐĂNG KÝ HỌC PHẦN THEO MÔN
# ==============================================================
class CourseEnrollment(Base):
    __tablename__ = "course_enrollments"
    id = Column(Integer, primary_key=True, index=True)
    subject_name = Column(String(100), index=True, nullable=False)
    student_id = Column(String(50), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==============================================================
# BẢNG NHẬT KÝ ĐIỂM DANH
# ==============================================================
class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), nullable=False)
    room_name = Column(String(50), index=True, nullable=False)
    student_id = Column(String(50), index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    subject_name = Column(String(100), nullable=False)
    similarity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(50), nullable=False)
    snapshot_path = Column(String(255), nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
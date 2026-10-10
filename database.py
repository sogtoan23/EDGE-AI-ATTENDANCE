import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Time, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Cấu hình chuỗi kết nối CSDL (Ưu tiên PostgreSQL trên Render, fallback SQLite khi chạy cục bộ)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render thường cung cấp URL dạng postgres://, cần đổi thành postgresql:// để SQLAlchemy nhận diện driver
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
else:
    # Chạy trên máy cục bộ với SQLite
    engine = create_engine(
        "sqlite:///./face_attendance.db",
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Định nghĩa các bảng CSDL
class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    class_name = Column(String(50), nullable=False)
    password_hash = Column(String(255), nullable=True)
    embedding_json = Column(Text, nullable=True)  # Vector 512-D mã hóa AES
    created_at = Column(DateTime, default=datetime.now)

class ClassSchedule(Base):
    __tablename__ = "class_schedules"

    id = Column(Integer, primary_key=True, index=True)
    room_name = Column(String(50), index=True, nullable=False)
    class_name = Column(String(50), nullable=True)
    subject_name = Column(String(100), nullable=False)
    teacher_name = Column(String(100), nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0: Thứ 2 ... 6: Chủ Nhật
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    early_arrival_minutes = Column(Integer, default=30)
    grace_period_minutes = Column(Integer, default=15)

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), nullable=False)
    room_name = Column(String(50), nullable=False)
    student_id = Column(String(50), index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    subject_name = Column(String(100), nullable=False)
    similarity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    status = Column(String(50), nullable=False)
    snapshot_path = Column(String(255), nullable=True)

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), unique=True, index=True, nullable=False)
    device_name = Column(String(100), nullable=True)
    room_name = Column(String(50), nullable=False)
    device_token = Column(String(100), unique=True, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)

class CourseEnrollment(Base):
    __tablename__ = "course_enrollments"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), index=True, nullable=False)
    subject_name = Column(String(100), nullable=False)

# Khởi tạo bảng tự động
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
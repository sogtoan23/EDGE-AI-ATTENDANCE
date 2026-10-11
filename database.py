import os
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    inspect,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./attendance.db")

# Render thường cung cấp DATABASE_URL bắt đầu bằng "postgres://" hoặc "postgresql://"
# Cần chuẩn hóa sang "postgresql+psycopg2://" để ép SQLAlchemy dùng driver psycopg2-binary đã cài đặt
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    class_name = Column(String(50), index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    embedding_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, index=True)
    schedule_id = Column(Integer, index=True, nullable=True)
    student_id = Column(String(50), index=True, nullable=False)
    room_name = Column(String(50), nullable=False)
    subject_name = Column(String(100), nullable=False)
    punch_time = Column(DateTime, default=datetime.now, index=True)
    status = Column(String(50), nullable=False)  # "Đúng giờ", "Đi trễ", "Vắng"
    similarity = Column(Float, default=0.0)
    snapshot_image = Column(Text, nullable=True)


class CameraDevice(Base):
    __tablename__ = "camera_devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(50), unique=True, index=True, nullable=False)
    device_name = Column(String(100), nullable=False)
    room_name = Column(String(50), index=True, nullable=False)
    device_token = Column(String(100), unique=True, index=True, nullable=False)
    webhook_url = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(Integer, primary_key=True, index=True)
    room_name = Column(String(50), index=True, nullable=False)
    class_name = Column(String(50), index=True, nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0: Thứ 2 -> 6: Chủ nhật
    subject_name = Column(String(100), nullable=False)
    teacher_name = Column(String(100), nullable=False)
    start_date = Column(String(10), nullable=True)  # "YYYY-MM-DD"
    end_date = Column(String(10), nullable=True)    # "YYYY-MM-DD"
    start_time = Column(String(10), nullable=False)  # "HH:MM"
    end_time = Column(String(10), nullable=False)    # "HH:MM"
    grace_period = Column(Integer, default=15)       # Mặc định 15 phút


def init_db_with_migrations():
    """Tự động kiểm tra và đồng bộ các cột mới cho PostgreSQL / SQLite"""
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    
    # 1. Đồng bộ bảng attendance_logs
    if "attendance_logs" in table_names:
        cols_log = [c["name"] for c in inspector.get_columns("attendance_logs")]
        with engine.connect() as conn:
            # Sửa lỗi: psycopg2.errors.UndefinedColumn: column attendance_logs.punch_time does not exist
            if "punch_time" not in cols_log:
                if "timestamp" in cols_log:
                    conn.execute(text("ALTER TABLE attendance_logs RENAME COLUMN timestamp TO punch_time;"))
                    print("✅ [DATABASE] Đã đổi tên cột timestamp -> punch_time")
                else:
                    conn.execute(text("ALTER TABLE attendance_logs ADD COLUMN punch_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP;"))
                    print("✅ [DATABASE] Đã thêm cột punch_time vào attendance_logs")

            if "schedule_id" not in cols_log:
                conn.execute(text("ALTER TABLE attendance_logs ADD COLUMN schedule_id INTEGER;"))
                print("✅ [DATABASE] Đã thêm cột schedule_id vào attendance_logs")

            conn.commit()

    # 2. Đồng bộ bảng schedules
    if "schedules" in table_names:
        cols_sch = [c["name"] for c in inspector.get_columns("schedules")]
        with engine.connect() as conn:
            if "start_date" not in cols_sch:
                conn.execute(text("ALTER TABLE schedules ADD COLUMN start_date VARCHAR(10);"))
            if "end_date" not in cols_sch:
                conn.execute(text("ALTER TABLE schedules ADD COLUMN end_date VARCHAR(10);"))
            conn.commit()
        print("✅ [DATABASE] Đồng bộ cột start_date, end_date cho bảng schedules thành công")


init_db_with_migrations()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
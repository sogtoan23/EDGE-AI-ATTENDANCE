import os
import json
import base64
import logging
import asyncio
import secrets
from io import BytesIO
from logging.handlers import RotatingFileHandler
from datetime import datetime, date, timedelta
from typing import List, Optional
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Depends, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException, Header, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from database import get_db, Student, ClassSchedule, AttendanceLog, Device, CourseEnrollment
from feature_extractor import FullPipelineFaceExtractor
from security import (
    verify_password, hash_password, create_access_token, decode_token,
    get_current_admin, get_current_student, security_bearer,
    encrypt_vector_string, decrypt_vector_string, ADMIN_PASSWORD_HASH
)

# -------------------------------------------------------------
# 1. CẤU HÌNH LOGGING & THƯ MỤC HỆ THỐNG
# -------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
os.makedirs("snapshots", exist_ok=True)
os.makedirs("static", exist_ok=True)

logger = logging.getLogger("FaceAttendanceApp")
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("logs/server.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

# -------------------------------------------------------------
# 2. KHỞI TẠO ỨNG DỤNG FASTAPI & TÁC VỤ NGẦM
# -------------------------------------------------------------
app = FastAPI(title="Face Attendance Enterprise Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

extractor = FullPipelineFaceExtractor()

async def cleanup_old_snapshots_task():
    """Tác vụ ngầm dọn dẹp các bức ảnh snapshot cũ hơn 30 ngày."""
    while True:
        try:
            now = datetime.now()
            cutoff = now - timedelta(days=30)
            count = 0
            for fname in os.listdir("snapshots"):
                fpath = os.path.join("snapshots", fname)
                if os.path.isfile(fpath):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
                    if file_mtime < cutoff:
                        os.remove(fpath)
                        count += 1
            if count > 0:
                logger.info(f"[CLEANUP] Đã dọn dẹp {count} ảnh snapshot cũ.")
        except Exception as e:
            logger.error(f"[CLEANUP ERROR] {e}")
        await asyncio.sleep(86400)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_old_snapshots_task())

class ConnectionManager:
    def __init__(self):
        self.active_websockets: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.append(websocket)
        logger.info(f"WebSocket client kết nối. Tổng: {len(self.active_websockets)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)
            logger.info(f"WebSocket client ngắt kết nối. Còn lại: {len(self.active_websockets)}")

    async def broadcast(self, message: dict):
        for connection in list(self.active_websockets):
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

# -------------------------------------------------------------
# 3. XÁC THỰC CAMERA DEVICE (X-Device-Token)
# -------------------------------------------------------------
def get_current_device(
    x_device_token: str = Header(..., alias="X-Device-Token"),
    db: Session = Depends(get_db)
) -> Device:
    device = db.query(Device).filter(
        Device.device_token == x_device_token,
        Device.is_active == 1
    ).first()

    if not device:
        logger.warning(f"Từ chối thiết bị camera với token: {x_device_token}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Thiết bị camera không hợp lệ hoặc đã bị vô hiệu hóa!"
        )
    return device

# -------------------------------------------------------------
# 4. XÁC THỰC NGƯỜI DÙNG & TRA CỨU ĐIỂM DANH
# -------------------------------------------------------------
class LoginPayload(BaseModel):
    username: str
    password: str

@app.post("/api/v1/auth/login")
def admin_login(payload: LoginPayload):
    """Đăng nhập dành cho Giảng Viên / Quản trị viên."""
    if payload.username.strip().lower() == "admin" and verify_password(payload.password, ADMIN_PASSWORD_HASH):
        token = create_access_token({"sub": "admin", "role": "admin"})
        logger.info("Giảng viên (admin) đăng nhập thành công.")
        return {"access_token": token, "token_type": "bearer", "status": "success", "role": "admin"}
    logger.warning(f"Giảng viên đăng nhập thất bại: {payload.username}")
    raise HTTPException(status_code=401, detail="Tài khoản hoặc mật khẩu quản trị không chính xác!")

@app.post("/api/v1/auth/student-login")
def student_login(payload: LoginPayload, db: Session = Depends(get_db)):
    """Đăng nhập dành cho Sinh Viên."""
    sid = payload.username.strip().upper()
    student = db.query(Student).filter(Student.student_id == sid).first()
    if not student:
        raise HTTPException(status_code=401, detail="Mã sinh viên không tồn tại trong hệ thống!")

    if not student.password_hash:
        student.password_hash = hash_password(sid)
        db.commit()

    if not verify_password(payload.password, student.password_hash):
        raise HTTPException(status_code=401, detail="Mật khẩu sinh viên không chính xác!")

    token = create_access_token({"sub": sid, "role": "student", "name": student.full_name})
    logger.info(f"Sinh viên {student.full_name} ({sid}) đăng nhập thành công.")
    return {
        "access_token": token,
        "token_type": "bearer",
        "student_id": student.student_id,
        "full_name": student.full_name,
        "class_name": student.class_name,
        "role": "student"
    }

@app.get("/api/v1/students/lookup-attendance")
def lookup_student_attendance(
    student_id: Optional[str] = None,
    auth: HTTPAuthorizationCredentials = Depends(security_bearer),
    db: Session = Depends(get_db)
):
    """Tra cứu điểm danh: Sinh viên xem chính mình, Giảng viên tra cứu sinh viên bất kỳ."""
    if not auth:
        raise HTTPException(status_code=401, detail="Yêu cầu đăng nhập để tra cứu dữ liệu!")
    payload = decode_token(auth.credentials)
    role = payload.get("role")

    if role == "student":
        target_sid = payload.get("sub")
    elif role == "admin":
        if not student_id:
            raise HTTPException(status_code=400, detail="Vui lòng cung cấp mã sinh viên cần tra cứu!")
        target_sid = student_id.strip().upper()
    else:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập!")

    student = db.query(Student).filter(Student.student_id == target_sid).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sinh viên có mã '{target_sid}'!")

    logs = db.query(AttendanceLog).filter(AttendanceLog.student_id == target_sid).order_by(AttendanceLog.timestamp.desc()).all()
    return {
        "student_id": student.student_id,
        "full_name": student.full_name,
        "class_name": student.class_name,
        "logs": [{
            "time": l.timestamp.strftime("%d/%m/%Y %H:%M:%S"),
            "room_name": l.room_name,
            "subject_name": l.subject_name,
            "status": l.status,
            "snapshot": l.snapshot_path
        } for l in logs]
    }

# -------------------------------------------------------------
# 5. TIẾP NHẬN ĐIỂM DANH TỪ CAMERA EDGE
# -------------------------------------------------------------
class EdgeAttendancePayload(BaseModel):
    student_id: str
    similarity: float
    timestamp: Optional[datetime] = None
    image_base64: Optional[str] = None

def process_single_attendance(payload: EdgeAttendancePayload, current_device: Device, db: Session) -> dict:
    check_time = payload.timestamp if payload.timestamp else datetime.now()
    cur_time = check_time.time()
    cur_weekday = check_time.weekday()
    cur_minutes = cur_time.hour * 60 + cur_time.minute

    target_room = current_device.room_name
    device_id = current_device.device_id
    target_sid = payload.student_id.strip().upper()

    student = db.query(Student).filter(Student.student_id == target_sid).first()
    full_name = student.full_name if student else "Unknown"
    student_class = student.class_name if student else "Unknown"

    schedules = db.query(ClassSchedule).filter(
        ClassSchedule.room_name == target_room,
        ClassSchedule.day_of_week == cur_weekday
    ).all()

    status_str = "Ngoài giờ học"
    subject_name = "Tự do / Ngoài ca"

    for sch in schedules:
        start_min = sch.start_time.hour * 60 + sch.start_time.minute
        end_min = sch.end_time.hour * 60 + sch.end_time.minute
        early_limit = start_min - getattr(sch, 'early_arrival_minutes', 30)
        late_limit = start_min + sch.grace_period_minutes

        if early_limit <= cur_minutes <= end_min:
            subject_name = sch.subject_name
            target_class = getattr(sch, 'class_name', '').strip()

            if target_class and student_class != target_class:
                status_str = f"Không thuộc lớp (Lớp {student_class})"
            elif cur_minutes <= late_limit:
                status_str = "Đúng giờ"
            else:
                status_str = "Đi trễ"
            break

    # Debounce 10 phút chống gửi trùng lặp dữ liệu[cite: 1]
    cooldown_limit = check_time - timedelta(minutes=10)
    recent_log = db.query(AttendanceLog).filter(
        AttendanceLog.student_id == target_sid,
        AttendanceLog.subject_name == subject_name,
        AttendanceLog.timestamp >= cooldown_limit,
        AttendanceLog.timestamp <= check_time
    ).first()

    if recent_log:
        return {
            "is_duplicate": True,
            "status": recent_log.status,
            "subject": subject_name,
            "full_name": full_name
        }

    snapshot_filename = None
    if payload.image_base64:
        try:
            img_data = base64.b64decode(payload.image_base64)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                if w > 480:
                    new_h = int(h * (480 / w))
                    img = cv2.resize(img, (480, new_h))

                filename = f"snap_{target_sid}_{int(check_time.timestamp())}.jpg"
                file_path = os.path.join("snapshots", filename)
                cv2.imwrite(file_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                snapshot_filename = f"/snapshots/{filename}"
        except Exception as e:
            logger.error(f"Lỗi nén ảnh snapshot: {e}")

    log = AttendanceLog(
        device_id=device_id,
        room_name=target_room,
        student_id=target_sid,
        full_name=full_name,
        subject_name=subject_name,
        similarity=payload.similarity,
        timestamp=check_time,
        status=status_str,
        snapshot_path=snapshot_filename
    )
    db.add(log)
    db.commit()

    return {
        "is_duplicate": False,
        "room_name": target_room,
        "subject_name": subject_name,
        "student_id": target_sid,
        "full_name": full_name,
        "time": check_time.strftime("%H:%M:%S"),
        "status": status_str,
        "similarity": round(payload.similarity, 2),
        "snapshot": snapshot_filename
    }

@app.post("/api/v1/attendance")
async def receive_attendance(
    payload: EdgeAttendancePayload,
    current_device: Device = Depends(get_current_device),
    db: Session = Depends(get_db)
):
    res = process_single_attendance(payload, current_device, db)
    if res["is_duplicate"]:
        return {"status": "ignored", "message": "Sinh viên vừa điểm danh trong vòng 10 phút trước."}

    logger.info(f"[ATTENDANCE] {res['full_name']} ({res['student_id']}) -> Phòng {res['room_name']}: {res['status']}")
    await manager.broadcast({"event": "NEW_ATTENDANCE", "data": res})
    return {
        "status": "success",
        "result_status": res["status"],
        "subject": res["subject_name"],
        "room": res["room_name"]
    }

@app.post("/api/v1/attendance/bulk-sync")
async def receive_bulk_attendance(
    payloads: List[EdgeAttendancePayload],
    current_device: Device = Depends(get_current_device),
    db: Session = Depends(get_db)
):
    processed = 0
    ignored = 0
    for item in payloads:
        res = process_single_attendance(item, current_device, db)
        if res["is_duplicate"]:
            ignored += 1
        else:
            processed += 1
            await manager.broadcast({"event": "NEW_ATTENDANCE", "data": res})

    return {"status": "success", "room": current_device.room_name, "inserted": processed, "ignored": ignored}

# -------------------------------------------------------------
# 6. ĐỒNG BỘ VECTOR CHO CAMERA EDGE
# -------------------------------------------------------------
@app.get("/api/v1/students/sync-all")
def sync_all(
    current_device: Device = Depends(get_current_device),
    db: Session = Depends(get_db)
):
    students = db.query(Student).all()
    results = []
    for s in students:
        if not s.embedding_json:
            continue
        try:
            raw_vec = json.loads(decrypt_vector_string(s.embedding_json))
            results.append({
                "student_id": s.student_id,
                "full_name": s.full_name,
                "embedding": raw_vec
            })
        except Exception as e:
            logger.error(f"Lỗi giải mã vector sinh viên {s.student_id}: {e}")
    return results

# -------------------------------------------------------------
# 7. QUẢN LÝ THIẾT BỊ CAMERA
# -------------------------------------------------------------
class RegisterDevicePayload(BaseModel):
    device_id: str
    device_name: str
    room_name: str

@app.post("/api/v1/devices/register", dependencies=[Depends(get_current_admin)])
def register_new_device(payload: RegisterDevicePayload, db: Session = Depends(get_db)):
    dev_id = payload.device_id.strip().upper()
    existing = db.query(Device).filter(Device.device_id == dev_id).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Mã camera '{dev_id}' đã tồn tại!")

    generated_token = f"DEV_{secrets.token_hex(16)}"
    new_device = Device(
        device_id=dev_id,
        device_name=payload.device_name.strip(),
        room_name=payload.room_name.strip().upper(),
        device_token=generated_token,
        is_active=1
    )
    db.add(new_device)
    db.commit()

    return {
        "status": "success",
        "message": f"Đã đăng ký camera cho phòng {new_device.room_name}",
        "device_token": generated_token
    }

@app.get("/api/v1/devices/list", dependencies=[Depends(get_current_admin)])
def list_devices(db: Session = Depends(get_db)):
    devices = db.query(Device).all()
    return [{
        "id": d.id,
        "device_id": d.device_id,
        "device_name": d.device_name,
        "room_name": d.room_name,
        "device_token": d.device_token,
        "is_active": d.is_active,
        "created_at": d.created_at.strftime("%d/%m/%Y %H:%M") if d.created_at else "N/A"
    } for d in devices]

@app.delete("/api/v1/devices/{device_id}", dependencies=[Depends(get_current_admin)])
def delete_device(device_id: str, db: Session = Depends(get_db)):
    dev = db.query(Device).filter(Device.device_id == device_id.strip().upper()).first()
    if not dev:
        raise HTTPException(status_code=404, detail="Không tìm thấy thiết bị cần xóa.")
    db.delete(dev)
    db.commit()
    return {"status": "success", "message": f"Đã xóa thiết bị {device_id}"}

# -------------------------------------------------------------
# 8. QUẢN LÝ EKYC (CHẶN MẶT GIẢ/ĐIỆN THOẠI & LƯU VĨNH VIỄN)
# -------------------------------------------------------------
@app.get("/api/v1/students/check/{student_id}")
def check_student_exists(student_id: str, db: Session = Depends(get_db)):
    sid = student_id.strip().upper()
    student = db.query(Student).filter(Student.student_id == sid).first()
    if student:
        return {"exists": True, "full_name": student.full_name, "class_name": student.class_name}
    return {"exists": False}

@app.get("/api/v1/students/list")
def list_students(class_name: Optional[str] = None, search: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Student)
    if class_name and class_name.strip() and class_name != "ALL":
        query = query.filter(Student.class_name == class_name.strip())
    if search and search.strip():
        search_kw = f"%{search.strip().upper()}%"
        query = query.filter(
            (Student.student_id.like(search_kw)) | (Student.full_name.ilike(f"%{search.strip()}%"))
        )

    students = query.order_by(Student.created_at.desc()).all()
    all_classes = [c[0] for c in db.query(Student.class_name).distinct().all() if c[0]]

    student_list = []
    for s in students:
        has_vector = False
        if s.embedding_json:
            try:
                decrypted = decrypt_vector_string(s.embedding_json)
                emb = json.loads(decrypted)
                if isinstance(emb, list) and len(emb) == 512:
                    has_vector = True
            except Exception:
                has_vector = False

        student_list.append({
            "id": s.id,
            "student_id": s.student_id,
            "full_name": s.full_name,
            "class_name": s.class_name,
            "has_vector": has_vector,
            "created_at": s.created_at.strftime("%d/%m/%Y %H:%M") if s.created_at else "N/A"
        })

    return {"classes": sorted(all_classes), "students": student_list}

@app.post("/api/v1/students/enroll-multi", dependencies=[Depends(get_current_admin)])
async def enroll_multi_faces(
    student_id: str = Form(...),
    full_name: str = Form(...),
    class_name: str = Form(...),
    overwrite: bool = Form(False),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    student_id = student_id.strip().upper()
    existing_student = db.query(Student).filter(Student.student_id == student_id).first()

    if existing_student and not overwrite:
        raise HTTPException(status_code=409, detail=f"Mã sinh viên {student_id} đã tồn tại trong hệ thống!")

    if len(photos) < 2:
        raise HTTPException(status_code=400, detail="Cần tối thiểu 2 góc ảnh để trích xuất vector chuẩn xác.")

    photos_bytes = [await p.read() for p in photos]

    # Duyệt tuần tự qua đủ 3 model AI
    try:
        embedding, liveness = extractor.process_multiple_photos(photos_bytes)
    except ValueError as ve:
        # Nếu model 1 (không thấy mặt) hoặc model 2 (phát hiện màn hình điện thoại) -> Chặn đứng quy trình ngay
        logger.warning(f"[EKYC BLOCKED] {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[AI ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi AI trích xuất vector: {str(e)}")

    # Mã hóa vector 512-D với AES-256
    encrypted_vector = encrypt_vector_string(json.dumps(embedding))

    if existing_student:
        existing_student.full_name = full_name.strip()
        existing_student.class_name = class_name.strip()
        existing_student.embedding_json = encrypted_vector
        action_type = "UPDATE_FACE"
        msg = f"Đã cập nhật vector khuôn mặt mới cho {full_name}!"
    else:
        new_student = Student(
            student_id=student_id,
            full_name=full_name.strip(),
            class_name=class_name.strip(),
            password_hash=hash_password(student_id),
            embedding_json=encrypted_vector
        )
        db.add(new_student)
        action_type = "ADD_FACE"
        msg = f"Đăng ký thành công {full_name} (Độ tin cậy người thật: {liveness*100:.1f}%)"

    db.commit()

    # Đồng bộ realtime xuống các Camera Edge
    await manager.broadcast({
        "event": "SYNC_LOCAL_CACHE",
        "action": action_type,
        "student_id": student_id,
        "full_name": full_name,
        "embedding": embedding
    })
    return {"status": "success", "message": msg}

class DeleteStudentPayload(BaseModel):
    student_id: str
    password: str

@app.post("/api/v1/students/delete", dependencies=[Depends(get_current_admin)])
async def delete_student(payload: DeleteStudentPayload, db: Session = Depends(get_db)):
    if not verify_password(payload.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=403, detail="Mật khẩu quản trị không chính xác!")

    sid = payload.student_id.strip().upper()
    student = db.query(Student).filter(Student.student_id == sid).first()
    if not student:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên.")

    full_name = student.full_name
    db.delete(student)
    db.commit()

    await manager.broadcast({
        "event": "SYNC_LOCAL_CACHE",
        "action": "DELETE_FACE",
        "student_id": sid
    })
    return {"status": "success", "message": f"Đã xóa thành công sinh viên {full_name} ({sid})"}

# -------------------------------------------------------------
# 9. LỊCH HỌC, TRA CỨU MA TRẬN & XUẤT EXCEL CHỐNG LỖI LATIN-1
# -------------------------------------------------------------
@app.get("/api/v1/schedules/list")
def get_schedules(db: Session = Depends(get_db)):
    days = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"]
    schedules = db.query(ClassSchedule).all()
    return [{
        "id": s.id,
        "room_name": s.room_name,
        "class_name": getattr(s, 'class_name', '') or '',
        "subject_name": s.subject_name,
        "teacher_name": s.teacher_name,
        "day_of_week": s.day_of_week,
        "day_text": days[s.day_of_week],
        "start_time": s.start_time.strftime("%H:%M"),
        "end_time": s.end_time.strftime("%H:%M"),
        "time": f"{s.start_time.strftime('%H:%M')} - {s.end_time.strftime('%H:%M')}",
        "early_arrival": getattr(s, 'early_arrival_minutes', 30),
        "grace": s.grace_period_minutes
    } for s in schedules]

@app.post("/api/v1/schedules", dependencies=[Depends(get_current_admin)])
def add_schedule(
    room_name: str = Form(...),
    class_name: str = Form(...),
    subject_name: str = Form(...),
    teacher_name: str = Form(...),
    day_of_week: int = Form(...),
    start_time_str: str = Form(...),
    end_time_str: str = Form(...),
    early_arrival: int = Form(30),
    grace_period: int = Form(15),
    db: Session = Depends(get_db)
):
    try:
        start_t = datetime.strptime(start_time_str, "%H:%M").time()
        end_t = datetime.strptime(end_time_str, "%H:%M").time()
    except Exception:
        raise HTTPException(status_code=400, detail="Định dạng thời gian không hợp lệ (HH:MM)")

    schedule = ClassSchedule(
        room_name=room_name.strip().upper(),
        class_name=class_name.strip(),
        subject_name=subject_name.strip(),
        teacher_name=teacher_name.strip(),
        day_of_week=day_of_week,
        start_time=start_t,
        end_time=end_t,
        early_arrival_minutes=early_arrival,
        grace_period_minutes=grace_period
    )
    db.add(schedule)
    db.commit()
    return {"status": "success", "message": f"Đã lưu lịch học môn {subject_name} cho lớp {class_name}"}

@app.put("/api/v1/schedules/{schedule_id}", dependencies=[Depends(get_current_admin)])
def update_schedule(
    schedule_id: int,
    room_name: str = Form(...),
    class_name: str = Form(...),
    subject_name: str = Form(...),
    teacher_name: str = Form(...),
    day_of_week: int = Form(...),
    start_time_str: str = Form(...),
    end_time_str: str = Form(...),
    early_arrival: int = Form(30),
    grace_period: int = Form(15),
    db: Session = Depends(get_db)
):
    sch = db.query(ClassSchedule).filter(ClassSchedule.id == schedule_id).first()
    if not sch:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch học cần sửa.")

    start_t = datetime.strptime(start_time_str, "%H:%M").time()
    end_t = datetime.strptime(end_time_str, "%H:%M").time()

    sch.room_name = room_name.strip().upper()
    sch.class_name = class_name.strip()
    sch.subject_name = subject_name.strip()
    sch.teacher_name = teacher_name.strip()
    sch.day_of_week = day_of_week
    sch.start_time = start_t
    sch.end_time = end_t
    sch.early_arrival_minutes = early_arrival
    sch.grace_period_minutes = grace_period
    db.commit()
    return {"status": "success", "message": "Đã cập nhật lịch học thành công!"}

@app.delete("/api/v1/schedules/{schedule_id}", dependencies=[Depends(get_current_admin)])
def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    sch = db.query(ClassSchedule).filter(ClassSchedule.id == schedule_id).first()
    if not sch:
        raise HTTPException(status_code=404, detail="Không tìm thấy ca học.")
    db.delete(sch)
    db.commit()
    return {"status": "success", "message": "Đã xóa ca học thành công!"}

@app.get("/api/v1/schedules/subjects")
def get_subjects(db: Session = Depends(get_db)):
    subjects = [s[0] for s in db.query(ClassSchedule.subject_name).distinct().all() if s[0]]
    logged_subjects = [s[0] for s in db.query(AttendanceLog.subject_name).distinct().all() if s[0]]
    return sorted(list(set(subjects + logged_subjects)))

@app.get("/api/v1/attendance/preview-by-class", dependencies=[Depends(get_current_admin)])
def preview_attendance_by_class(
    class_name: str,
    subject_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    class_name = class_name.strip()
    subject_name = subject_name.strip()

    students = db.query(Student).filter(Student.class_name == class_name).order_by(Student.student_id).all()
    if not students:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sinh viên nào thuộc lớp '{class_name}'.")

    log_query = db.query(AttendanceLog).filter(AttendanceLog.subject_name == subject_name)
    if start_date:
        try:
            s_dt = datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.min.time())
            log_query = log_query.filter(AttendanceLog.timestamp >= s_dt)
        except Exception:
            pass
    if end_date:
        try:
            e_dt = datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), datetime.max.time())
            log_query = log_query.filter(AttendanceLog.timestamp <= e_dt)
        except Exception:
            pass

    logs = log_query.all()
    unique_dates = sorted(list(set(log.timestamp.strftime("%d/%m/%Y") for log in logs)))

    attendance_map = {}
    for log in logs:
        date_key = log.timestamp.strftime("%d/%m/%Y")
        time_key = log.timestamp.strftime("%H:%M")
        map_key = (log.student_id, date_key)
        if map_key not in attendance_map or "Đúng giờ" in log.status:
            attendance_map[map_key] = f"{log.status} ({time_key})"

    records = []
    for idx, stu in enumerate(students):
        item = {
            "stt": idx + 1,
            "student_id": stu.student_id,
            "full_name": stu.full_name,
            "class_name": stu.class_name,
            "dates": {},
            "summary": {"dung": 0, "tre": 0, "vang": 0}
        }
        for d in unique_dates:
            status_text = attendance_map.get((stu.student_id, d), "Vắng")
            item["dates"][d] = status_text
            if "Đúng giờ" in status_text:
                item["summary"]["dung"] += 1
            elif "Đi trễ" in status_text:
                item["summary"]["tre"] += 1
            else:
                item["summary"]["vang"] += 1
        records.append(item)

    return {
        "class_name": class_name,
        "subject_name": subject_name,
        "dates": unique_dates,
        "records": records
    }

@app.get("/api/v1/attendance/export-by-class", dependencies=[Depends(get_current_admin)])
def export_attendance_by_class(
    class_name: str,
    subject_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    try:
        class_name = class_name.strip()
        subject_name = subject_name.strip()

        students = db.query(Student).filter(Student.class_name == class_name).order_by(Student.student_id).all()
        if not students:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy sinh viên nào thuộc lớp '{class_name}'.")

        log_query = db.query(AttendanceLog).filter(AttendanceLog.subject_name == subject_name)
        if start_date:
            try:
                s_dt = datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), datetime.min.time())
                log_query = log_query.filter(AttendanceLog.timestamp >= s_dt)
            except Exception:
                pass
        if end_date:
            try:
                e_dt = datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), datetime.max.time())
                log_query = log_query.filter(AttendanceLog.timestamp <= e_dt)
            except Exception:
                pass

        logs = log_query.all()
        unique_dates = sorted(list(set(log.timestamp.strftime("%d/%m/%Y") for log in logs)))

        attendance_map = {}
        for log in logs:
            date_key = log.timestamp.strftime("%d/%m/%Y")
            time_key = log.timestamp.strftime("%H:%M")
            map_key = (log.student_id, date_key)
            if map_key not in attendance_map or "Đúng giờ" in log.status:
                attendance_map[map_key] = f"{log.status} ({time_key})"

        rows = []
        for idx, stu in enumerate(students):
            row = {
                "STT": idx + 1,
                "Mã Sinh Viên": stu.student_id,
                "Họ và Tên": stu.full_name,
                "Lớp": stu.class_name
            }
            dung, tre, vang = 0, 0, 0
            for d in unique_dates:
                status_text = attendance_map.get((stu.student_id, d), "Vắng")
                row[f"Ngày {d}"] = status_text
                if "Đúng giờ" in status_text:
                    dung += 1
                elif "Đi trễ" in status_text:
                    tre += 1
                else:
                    vang += 1
            row["Tổng Đúng Giờ"] = dung
            row["Tổng Đi Trễ"] = tre
            row["Tổng Vắng"] = vang
            rows.append(row)

        df = pd.DataFrame(rows)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            safe_sheet = f"{class_name[:10]}_{subject_name[:10]}"
            df.to_excel(writer, index=False, sheet_name=safe_sheet)

        output.seek(0)

        # Chuẩn RFC 5987 chống lỗi 'latin-1' codec khi tên file có dấu tiếng Việt
        clean_class = "".join(c for c in class_name if c.isalnum() or c in (' ', '_', '-')).strip()
        clean_subj = "".join(c for c in subject_name if c.isalnum() or c in (' ', '_', '-')).strip()
        date_str = datetime.now().strftime('%Y%m%d')

        filename_utf8 = f"DiemDanh_{clean_class}_{clean_subj}_{date_str}.xlsx"
        encoded_filename = quote(filename_utf8)
        fallback_filename = f"DiemDanh_Lop_{date_str}.xlsx"

        headers = {
            "Content-Disposition": f"attachment; filename=\"{fallback_filename}\"; filename*=UTF-8''{encoded_filename}"
        }

        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"[EXPORT ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi máy chủ khi tạo file Excel: {str(e)}")

# -------------------------------------------------------------
# 10. WEBSOCKET & INDEX HTML ROUTING
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/", response_class=HTMLResponse)
def index_view():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
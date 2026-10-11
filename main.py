import os
import json
import secrets
from datetime import datetime, timedelta
from typing import List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import (
    get_db,
    SessionLocal,
    Student,
    AttendanceLog,
    CameraDevice,
    Schedule,
    Base,
    engine
)

from security import (
    encrypt_vector_string,
    decrypt_vector_string,
    verify_password,
    hash_password,
    create_access_token,
    get_current_admin,
    ADMIN_PASSWORD_HASH
)

from feature_extractor import FullPipelineFaceExtractor

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Edge AI Attendance Gateway", 
    version="6.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = FullPipelineFaceExtractor()
DAY_NAMES = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


# --------------------------------------------------------------------------
# 1. QUẢN LÝ WEBSOCKET SERVER - EDGE & WEB DASHBOARD
# --------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


async def broadcast_edge_upsert(student_id: str, embedding_512: list):
    await manager.broadcast({
        "event": "SYNC_UPSERT_FACE",
        "data": {
            "student_id": student_id,
            "embedding": embedding_512
        }
    })

async def broadcast_edge_delete(student_id: str):
    await manager.broadcast({
        "event": "SYNC_DELETE_FACE",
        "data": {
            "student_id": student_id
        }
    })


# --------------------------------------------------------------------------
# 2. XÁC THỰC NGƯỜI DÙNG (AUTHENTICATION)
# --------------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/v1/auth/login")
def login_teacher(req: LoginRequest):
    if req.username == "admin" and verify_password(req.password, ADMIN_PASSWORD_HASH):
        token = create_access_token({"sub": "admin", "role": "admin"})
        return {"access_token": token, "token_type": "bearer", "role": "admin"}
    raise HTTPException(status_code=400, detail="Mật khẩu Giảng viên không đúng!")

@app.post("/api/v1/auth/student-login")
def login_student(req: LoginRequest, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_id == req.username).first()
    if not student:
        raise HTTPException(status_code=400, detail="Mã sinh viên không tồn tại!")
    if student.password_hash and not verify_password(req.password, student.password_hash):
        raise HTTPException(status_code=400, detail="Mật khẩu không đúng!")
    token = create_access_token({"sub": student.student_id, "role": "student"})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": "student",
        "student_id": student.student_id,
        "full_name": student.full_name,
        "class_name": student.class_name or "N/A"
    }


# --------------------------------------------------------------------------
# 3. ĐĂNG KÝ eKYC 3 GÓC
# --------------------------------------------------------------------------
@app.get("/api/v1/students/check/{student_id}")
def check_student_exists(student_id: str, db: Session = Depends(get_db)):
    s = db.query(Student).filter(Student.student_id == student_id).first()
    return {"exists": s is not None}

@app.post("/api/v1/students/enroll-multi")
async def enroll_multi_photos(
    student_id: str = Form(...),
    full_name: str = Form(...),
    class_name: str = Form(...),
    overwrite: bool = Form(False),
    photos: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if len(photos) < 3:
        raise HTTPException(status_code=400, detail="Cần cung cấp đủ 3 ảnh quét khuôn mặt!")

    existing = db.query(Student).filter(Student.student_id == student_id).first()
    if existing and not overwrite:
        raise HTTPException(status_code=400, detail=f"Mã sinh viên {student_id} đã tồn tại!")

    collected_embeddings = []
    for idx, photo in enumerate(photos, 1):
        contents = await photo.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail=f"Ảnh góc {idx} không hợp lệ!")

        bbox, landmarks, _ = extractor.detect_face(frame)
        if bbox is None:
            raise HTTPException(status_code=400, detail=f"Góc {idx}: Không tìm thấy khuôn mặt!")

        liveness_score = extractor.check_liveness(frame, bbox)
        if liveness_score < 0.70:
            raise HTTPException(
                status_code=400,
                detail=f"Góc {idx}: Phát hiện giả mạo màn hình hoặc ảnh in ({liveness_score:.2f} < 0.70)!"
            )

        aligned = extractor.align_face(frame, landmarks)
        embedding = extractor.extract_features(aligned)
        collected_embeddings.append(embedding)

    avg_vec = np.mean(collected_embeddings, axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm > 0:
        avg_vec = avg_vec / norm

    enc_vector = encrypt_vector_string(json.dumps(avg_vec.tolist()))
    clean_class = class_name.strip()

    if existing:
        existing.full_name = full_name.strip()
        existing.class_name = clean_class
        existing.embedding_json = enc_vector
    else:
        new_stu = Student(
            student_id=student_id.strip(),
            full_name=full_name.strip(),
            class_name=clean_class,
            password_hash=hash_password(student_id.strip()),
            embedding_json=enc_vector
        )
        db.add(new_stu)

    db.commit()
    await broadcast_edge_upsert(student_id.strip(), avg_vec.tolist())
    return {"status": "success", "message": f"Đã lưu sinh viên {full_name} và đồng bộ vector qua WebSocket!"}


# --------------------------------------------------------------------------
# 4. QUẢN LÝ SINH VIÊN & XÓA
# --------------------------------------------------------------------------
@app.get("/api/v1/students/list")
def list_students(
    class_name: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Student)
    if class_name and class_name != "ALL":
        query = query.filter(Student.class_name.ilike(f"%{class_name.strip()}%"))
    if search:
        search_kw = f"%{search.strip()}%"
        query = query.filter((Student.student_id.ilike(search_kw)) | (Student.full_name.ilike(search_kw)))

    students = query.order_by(Student.created_at.desc()).all()
    all_classes = [c[0] for c in db.query(Student.class_name).distinct().all() if c[0]]

    records = []
    for s in students:
        has_vec = False
        if s.embedding_json:
            try:
                decrypted = decrypt_vector_string(s.embedding_json)
                vec = json.loads(decrypted)
                if isinstance(vec, list) and len(vec) == 512:
                    has_vec = True
            except Exception:
                has_vec = False

        records.append({
            "student_id": s.student_id,
            "full_name": s.full_name,
            "class_name": s.class_name or "N/A",
            "created_at": s.created_at.strftime("%d/%m/%Y %H:%M") if s.created_at else "",
            "has_vector": has_vec
        })

    return {"classes": all_classes, "students": records}

class DeleteStudentRequest(BaseModel):
    student_id: str
    password: str

@app.post("/api/v1/students/delete")
async def delete_student_secured(req: DeleteStudentRequest, db: Session = Depends(get_db)):
    if not verify_password(req.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=400, detail="Mật khẩu quản trị không chính xác!")

    stu = db.query(Student).filter(Student.student_id == req.student_id).first()
    if not stu:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên!")

    sid = stu.student_id
    db.query(AttendanceLog).filter(AttendanceLog.student_id == sid).delete()
    db.delete(stu)
    db.commit()

    await broadcast_edge_delete(sid)
    return {"status": "success", "message": f"Đã xóa hoàn toàn sinh viên {sid} và phát lệnh gỡ bỏ khỏi Edge!"}


# --------------------------------------------------------------------------
# 5. ĐỒNG BỘ VECTOR CHO EDGE CAMERA
# --------------------------------------------------------------------------
@app.get("/api/v1/students/sync-vectors")
def sync_vectors_to_edge(
    x_device_token: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    if not x_device_token:
        raise HTTPException(status_code=401, detail="Thiếu mã xác thực thiết bị (X-Device-Token)!")

    camera = db.query(CameraDevice).filter(CameraDevice.device_token == x_device_token).first()
    if not camera:
        raise HTTPException(status_code=403, detail="Mã Camera không hợp lệ hoặc chưa được đăng ký!")

    students = db.query(Student).filter(Student.embedding_json.isnot(None)).all()
    payload = []
    for s in students:
        try:
            decrypted = decrypt_vector_string(s.embedding_json)
            vector_data = json.loads(decrypted)
            if isinstance(vector_data, list) and len(vector_data) == 512:
                payload.append({"student_id": s.student_id, "embedding": vector_data})
        except Exception:
            continue
    return payload

@app.post("/api/v1/edge/trigger-sync-all")
async def trigger_sync_all_devices(
    admin: dict = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    students = db.query(Student).filter(Student.embedding_json.isnot(None)).all()
    payload = []
    for s in students:
        try:
            decrypted = decrypt_vector_string(s.embedding_json)
            vector_data = json.loads(decrypted)
            if isinstance(vector_data, list) and len(vector_data) == 512:
                payload.append({"student_id": s.student_id, "embedding": vector_data})
        except Exception:
            continue

    await manager.broadcast({"event": "SYNC_FULL_GALLERY", "data": payload})
    return {"status": "success", "message": f"Đã phát lệnh nạp toàn bộ {len(payload)} vector qua WebSocket!"}


# --------------------------------------------------------------------------
# 6. TIẾP NHẬN ĐIỂM DANH: DYNAMIC ROOM BUFFER (TRỐNG PHÒNG MỞ CỬA SỚM 60 PHÚT)
# --------------------------------------------------------------------------
class AttendancePunchRequest(BaseModel):
    student_id: str
    similarity: float
    timestamp: Optional[str] = None
    image_base64: Optional[str] = None

@app.post("/api/v1/attendance/punch")
async def receive_punch(
    data: AttendancePunchRequest,
    x_device_token: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    if not x_device_token:
        raise HTTPException(status_code=401, detail="Thiếu mã token phòng học (X-Device-Token)!")

    camera = db.query(CameraDevice).filter(CameraDevice.device_token == x_device_token).first()
    if not camera:
        raise HTTPException(status_code=403, detail="Mã Camera/Phòng học không hợp lệ!")

    room_name = camera.room_name
    device_name = camera.device_name

    student = db.query(Student).filter(Student.student_id == data.student_id.strip()).first()
    if not student:
        raise HTTPException(status_code=404, detail=f"Mã sinh viên {data.student_id} không tồn tại!")

    student_class = student.class_name.strip()

    punch_time = datetime.fromisoformat(data.timestamp) if data.timestamp else datetime.now()
    punch_date_str = punch_time.strftime("%Y-%m-%d")
    weekday = punch_time.weekday()
    punch_hm = punch_time.strftime("%H:%M")
    today_start = punch_time.replace(hour=0, minute=0, second=0, microsecond=0)

    # 1. Lấy tất cả ca học tại phòng này trong ngày hôm nay, xếp theo thứ tự giờ bắt đầu
    room_schedules = db.query(Schedule).filter(
        Schedule.room_name == room_name,
        (Schedule.day_of_week == weekday) | (Schedule.day_of_week == str(weekday))
    ).order_by(Schedule.start_time).all()

    matched_schedule = None
    status_str = None

    for i, sch in enumerate(room_schedules):
        # Kiểm tra ngày môn học hợp lệ
        if sch.start_date and sch.end_date:
            if not (sch.start_date <= punch_date_str <= sch.end_date):
                continue

        # Chỉ so khớp nếu ca học thuộc lớp của sinh viên
        if sch.class_name.strip().lower() != student_class.lower():
            continue

        # DYNAMIC BUFFER: Xác định giờ mở cửa nhận diện sớm
        t_start = datetime.strptime(sch.start_time, "%H:%M")
        prev_sch = room_schedules[i - 1] if i > 0 else None

        if prev_sch:
            # Nếu trước ca này có lớp khác: Cho phép quét ngay khi lớp trước vừa tan
            t_early = datetime.strptime(prev_sch.end_time, "%H:%M")
        else:
            # Nếu trước ca này phòng trống: Mở nhận diện sớm trước 60 phút
            t_early = t_start - timedelta(minutes=60)

        early_hm = t_early.strftime("%H:%M")
        end_hm = sch.end_time
        
        # Mặc định ân hạn trễ 15 phút
        t_grace = t_start + timedelta(minutes=sch.grace_period or 15)
        grace_hm = t_grace.strftime("%H:%M")

        if early_hm <= punch_hm <= end_hm:
            matched_schedule = sch
            if punch_hm <= grace_hm:
                status_str = "Đúng giờ"
            else:
                status_str = "Đi trễ"
            break

    if not matched_schedule:
        print(f"⚠️ [BỎ QUA] SV {student.full_name} ({student.student_id}) lớp {student_class} không có ca học hợp lệ tại phòng {room_name} lúc {punch_hm}")
        return {
            "status": "ignored",
            "message": f"Sinh viên {student.full_name} (Lớp {student_class}) không có ca học tại phòng {room_name} lúc {punch_hm}. Bỏ qua!"
        }

    # Chống điểm danh trùng lặp trong ca học này
    existing_log = db.query(AttendanceLog).filter(
        AttendanceLog.schedule_id == matched_schedule.id,
        AttendanceLog.student_id == student.student_id,
        AttendanceLog.punch_time >= today_start
    ).first()

    if existing_log:
        return {
            "status": "ignored",
            "message": f"Sinh viên {student.full_name} đã điểm danh ca này rồi ({existing_log.status})."
        }

    log = AttendanceLog(
        schedule_id=matched_schedule.id,
        student_id=student.student_id,
        room_name=room_name,
        subject_name=matched_schedule.subject_name,
        punch_time=punch_time,
        status=status_str,
        similarity=round(data.similarity, 2),
        snapshot_image=data.image_base64
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    print(f"🎯 [ĐIỂM DANH THÀNH CÔNG] SV: {student.full_name} | Lớp: {student_class} | Ca ID: {matched_schedule.id} ({matched_schedule.subject_name}) tại {room_name} -> {status_str}")

    await manager.broadcast({
        "event": "NEW_ATTENDANCE",
        "data": {
            "room_name": room_name,
            "device_name": device_name,
            "subject_name": matched_schedule.subject_name,
            "student_id": student.student_id,
            "full_name": student.full_name,
            "class_name": student_class,
            "time": punch_time.strftime("%d/%m/%Y %H:%M:%S"),
            "status": status_str,
            "snapshot": log.snapshot_image
        }
    })

    return {
        "status": "success",
        "schedule_id": matched_schedule.id,
        "room_name": room_name,
        "subject_name": matched_schedule.subject_name,
        "student_id": student.student_id,
        "full_name": student.full_name,
        "class_name": student_class,
        "result_status": status_str
    }


# --------------------------------------------------------------------------
# 7. TRA CỨU ĐIỂM DANH CÁ NHÂN
# --------------------------------------------------------------------------
@app.get("/api/v1/students/lookup-attendance")
def lookup_student_attendance(
    student_id: Optional[str] = None,
    auth_header: Optional[str] = Header(None, alias="Authorization"),
    db: Session = Depends(get_db)
):
    target_sid = student_id
    if not target_sid and auth_header and "Bearer " in auth_header:
        from security import decode_token
        payload = decode_token(auth_header.split("Bearer ")[1])
        target_sid = payload.get("sub")

    stu = db.query(Student).filter(Student.student_id == target_sid).first()
    if not stu:
        raise HTTPException(status_code=404, detail="Không tìm thấy sinh viên!")

    logs = db.query(AttendanceLog).filter(
        AttendanceLog.student_id == stu.student_id
    ).order_by(AttendanceLog.punch_time.desc()).all()

    res_logs = []
    for l in logs:
        res_logs.append({
            "schedule_id": l.schedule_id,
            "time": l.punch_time.strftime("%d/%m/%Y %H:%M:%S") if l.punch_time else "",
            "room_name": l.room_name,
            "subject_name": l.subject_name,
            "status": l.status,
            "snapshot": l.snapshot_image
        })

    # Kiểm tra động các ca học hôm nay của sinh viên
    now = datetime.now()
    weekday = now.weekday()
    today_str = now.strftime("%Y-%m-%d")
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    current_hm = now.strftime("%H:%M")
    clean_class = stu.class_name.strip() if stu.class_name else ""

    today_schedules = db.query(Schedule).filter(
        (Schedule.day_of_week == weekday) | (Schedule.day_of_week == str(weekday)),
        Schedule.class_name.ilike(f"%{clean_class}%")
    ).all()

    for sch in today_schedules:
        if sch.start_date and sch.end_date:
            if not (sch.start_date <= today_str <= sch.end_date):
                continue

        # Ca học đã tan lớp mà chưa có log -> Báo Vắng[cite: 1, 4]
        if sch.end_time <= current_hm:
            attended = db.query(AttendanceLog).filter(
                AttendanceLog.schedule_id == sch.id,
                AttendanceLog.student_id == stu.student_id,
                AttendanceLog.punch_time >= today_start
            ).first()

            if not attended:
                res_logs.insert(0, {
                    "schedule_id": sch.id,
                    "time": f"{today_str} {sch.end_time}:00",
                    "room_name": sch.room_name,
                    "subject_name": sch.subject_name,
                    "status": "Vắng",
                    "snapshot": None
                })

    return {
        "student_id": stu.student_id,
        "full_name": stu.full_name,
        "class_name": stu.class_name,
        "logs": res_logs
    }


# --------------------------------------------------------------------------
# 8. XUẤT MA TRẬN BÁO CÁO: GIỚI HẠN THEO START_DATE & END_DATE MÔN HỌC
# --------------------------------------------------------------------------
@app.get("/api/v1/attendance/preview-by-class")
def preview_class_attendance(
    class_name: str,
    subject_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    admin: dict = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    clean_class = class_name.strip()
    students = db.query(Student).filter(
        Student.class_name.ilike(f"%{clean_class}%")
    ).order_by(Student.student_id).all()
    
    if not students:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy sinh viên thuộc lớp {class_name}")

    now = datetime.now()
    current_hm = now.strftime("%H:%M")
    today_d = now.date()

    # Lấy các ca học môn này của lớp
    schedules = db.query(Schedule).filter(
        Schedule.subject_name == subject_name,
        Schedule.class_name.ilike(f"%{clean_class}%")
    ).all()

    # Xác định khoảng ngày lọc
    if start_date:
        d_start = datetime.fromisoformat(start_date).date()
    else:
        earliest_start = min([sch.start_date for sch in schedules if sch.start_date], default=None)
        d_start = datetime.fromisoformat(earliest_start).date() if earliest_start else today_d

    if end_date:
        d_end = datetime.fromisoformat(end_date).date()
    else:
        latest_end = max([sch.end_date for sch in schedules if sch.end_date], default=None)
        d_end = min(today_d, datetime.fromisoformat(latest_end).date()) if latest_end else today_d

    actual_logs = db.query(AttendanceLog).filter(
        AttendanceLog.subject_name == subject_name,
        AttendanceLog.punch_time >= datetime.combine(d_start, datetime.min.time()),
        AttendanceLog.punch_time <= datetime.combine(d_end, datetime.max.time())
    ).all()

    log_map = {}
    for l in actual_logs:
        if l.punch_time:
            d_str = l.punch_time.strftime("%d/%m/%Y")
            log_map[(d_str, l.schedule_id, l.student_id)] = l.status

    sessions = []
    curr_d = d_start
    while curr_d <= d_end:
        w = curr_d.weekday()
        curr_d_str = curr_d.strftime("%Y-%m-%d")

        for sch in schedules:
            if sch.start_date and sch.end_date:
                if not (sch.start_date <= curr_d_str <= sch.end_date):
                    continue

            if int(sch.day_of_week) == w:
                is_finished = False
                if curr_d < today_d:
                    is_finished = True
                elif curr_d == today_d and sch.end_time <= current_hm:
                    is_finished = True

                if is_finished:
                    d_vn_str = curr_d.strftime("%d/%m/%Y")
                    sessions.append({
                        "date_str": d_vn_str,
                        "schedule_id": sch.id,
                        "col_header": f"{d_vn_str} ({sch.start_time}-{sch.end_time})"
                    })
        curr_d += timedelta(days=1)

    column_headers = [s["col_header"] for s in sessions]

    records = []
    for idx, s in enumerate(students, 1):
        stu_dates = {}
        dung = 0
        tre = 0
        vang = 0

        for ses in sessions:
            status = log_map.get((ses["date_str"], ses["schedule_id"], s.student_id))
            col_key = ses["col_header"]
            if status:
                stu_dates[col_key] = status
                if status == "Đúng giờ":
                    dung += 1
                elif status == "Đi trễ":
                    tre += 1
            else:
                stu_dates[col_key] = "Vắng"
                vang += 1

        records.append({
            "stt": idx,
            "student_id": s.student_id,
            "full_name": s.full_name,
            "class_name": s.class_name,
            "dates": stu_dates,
            "summary": {"dung": dung, "tre": tre, "vang": vang}
        })

    return {
        "class_name": class_name,
        "subject_name": subject_name,
        "dates": column_headers,
        "records": records
    }


# --------------------------------------------------------------------------
# 9. QUẢN LÝ THIẾT BỊ CAMERA & LỊCH HỌC (CRUD)
# --------------------------------------------------------------------------
@app.get("/api/v1/devices/list")
def list_devices(admin: dict = Depends(get_current_admin), db: Session = Depends(get_db)):
    return db.query(CameraDevice).all()

@app.post("/api/v1/devices/register")
def register_device(payload: dict, admin: dict = Depends(get_current_admin), db: Session = Depends(get_db)):
    token = "DEV_" + secrets.token_hex(16).upper()
    dev = CameraDevice(
        device_id=payload["device_id"],
        device_name=payload["device_name"],
        room_name=payload["room_name"],
        device_token=token,
        webhook_url=payload.get("webhook_url")
    )
    db.add(dev)
    db.commit()
    return {"status": "success", "message": "Đã thêm Camera thành công!", "device_token": token}

@app.delete("/api/v1/devices/{device_id}")
def delete_device(device_id: str, admin: dict = Depends(get_current_admin), db: Session = Depends(get_db)):
    dev = db.query(CameraDevice).filter(CameraDevice.device_id == device_id).first()
    if dev:
        db.query(AttendanceLog).filter(AttendanceLog.room_name == dev.room_name).delete()
        db.delete(dev)
        db.commit()
    return {"status": "success", "message": f"Đã xóa camera {device_id}"}

@app.get("/api/v1/schedules/list")
def list_schedules(db: Session = Depends(get_db)):
    schs = db.query(Schedule).order_by(Schedule.day_of_week, Schedule.start_time).all()
    results = []
    for s in schs:
        day_idx = int(s.day_of_week) if str(s.day_of_week).isdigit() else 0
        results.append({
            "id": s.id,
            "room_name": s.room_name,
            "class_name": s.class_name,
            "day_of_week": s.day_of_week,
            "day_text": DAY_NAMES[day_idx] if day_idx < len(DAY_NAMES) else f"Thứ {s.day_of_week}",
            "start_date": s.start_date or "",
            "end_date": s.end_date or "",
            "time": f"{s.start_time} - {s.end_time}",
            "start_time": s.start_time,
            "end_time": s.end_time,
            "subject_name": s.subject_name,
            "teacher_name": s.teacher_name,
            "grace": s.grace_period or 15
        })
    return results

@app.post("/api/v1/schedules")
def add_schedule(
    room_name: str = Form(...),
    class_name: str = Form(...),
    day_of_week: int = Form(...),
    subject_name: str = Form(...),
    teacher_name: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    start_time_str: str = Form(...),
    end_time_str: str = Form(...),
    grace_period: int = Form(15),
    admin: dict = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    sch = Schedule(
        room_name=room_name.strip(),
        class_name=class_name.strip(),
        day_of_week=day_of_week,
        subject_name=subject_name.strip(),
        teacher_name=teacher_name.strip(),
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        start_time=start_time_str,
        end_time=end_time_str,
        grace_period=grace_period
    )
    db.add(sch)
    db.commit()
    return {"status": "success", "message": "Đã thêm lịch học mới thành công!"}

@app.put("/api/v1/schedules/{id}")
def update_schedule(
    id: int,
    room_name: str = Form(...),
    class_name: str = Form(...),
    day_of_week: int = Form(...),
    subject_name: str = Form(...),
    teacher_name: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    start_time_str: str = Form(...),
    end_time_str: str = Form(...),
    grace_period: int = Form(15),
    admin: dict = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    sch = db.query(Schedule).filter(Schedule.id == id).first()
    if not sch:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch học cần cập nhật!")

    sch.room_name = room_name.strip()
    sch.class_name = class_name.strip()
    sch.day_of_week = day_of_week
    sch.subject_name = subject_name.strip()
    sch.teacher_name = teacher_name.strip()
    sch.start_date = start_date.strip()
    sch.end_date = end_date.strip()
    sch.start_time = start_time_str
    sch.end_time = end_time_str
    sch.grace_period = grace_period

    db.commit()
    db.refresh(sch)
    return {"status": "success", "message": "Đã cập nhật lịch học thành công!"}

@app.delete("/api/v1/schedules/{id}")
def delete_schedule(id: int, admin: dict = Depends(get_current_admin), db: Session = Depends(get_db)):
    sch = db.query(Schedule).filter(Schedule.id == id).first()
    if sch:
        db.query(AttendanceLog).filter(AttendanceLog.schedule_id == id).delete()
        db.delete(sch)
        db.commit()
    return {"status": "success", "message": "Đã xóa lịch học"}

@app.get("/api/v1/schedules/subjects")
def list_subjects(db: Session = Depends(get_db)):
    return [s[0] for s in db.query(Schedule.subject_name).distinct().all() if s[0]]

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    @app.get("/")
    def serve_ui():
        return FileResponse("static/index.html")
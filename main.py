import os
import re
import shutil
import secrets
import json
import time
import gc
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Safe Imports for PDF processing
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

# --- ROBUST GEMINI SDK HANDLER ---
ai_client_type = None
ai_client = None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        import google.generativeai as legacy_genai
        legacy_genai.configure(api_key=GEMINI_API_KEY)
        ai_client_type = "legacy"
    except ImportError:
        try:
            from google import genai
            from google.genai import types
            ai_client = genai.Client(api_key=GEMINI_API_KEY)
            ai_client_type = "new"
        except ImportError:
            print("WARNING: Gemini SDK not found.")

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, joinedload

# --- SECURITY ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")
USER1_USERNAME = os.getenv("USER1_USERNAME", "staff1")
USER1_PASSWORD = os.getenv("USER1_PASSWORD", "staff123")
USER2_USERNAME = os.getenv("USER2_USERNAME", "staff2")
USER2_PASSWORD = os.getenv("USER2_PASSWORD", "staff123")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    u, p = credentials.username, credentials.password
    if secrets.compare_digest(u, ADMIN_USERNAME) and secrets.compare_digest(p, ADMIN_PASSWORD): return {"username": u, "role": "admin"}
    if secrets.compare_digest(u, USER1_USERNAME) and secrets.compare_digest(p, USER1_PASSWORD): return {"username": u, "role": "user"}
    if secrets.compare_digest(u, USER2_USERNAME) and secrets.compare_digest(p, USER2_PASSWORD): return {"username": u, "role": "user"}
    raise HTTPException(status_code=401, detail="Invalid credentials", headers={"WWW-Authenticate": "Bearer"})

def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "admin": raise HTTPException(status_code=403, detail="Admin required.")
    return user

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- DATABASE MODELS ---
class Student(Base):
    __tablename__ = "students"
    registration_no = Column(String, primary_key=True, index=True)
    roll_no = Column(String)
    name = Column(String)
    admission_year = Column(String)
    passout_year = Column(String, default="Nil")
    curriculum = Column(String, default="Unknown") 
    course = Column(String, default="Unknown Course")
    subject = Column(String, default="Unknown")
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    remarks = Column(String, default="Qualified")
    
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    marksheet_issue_date = Column(String, default="")
    certificate_issue_date = Column(String, default="")
    issued_by = Column(String, default="")
    
    post_grad_status = Column(String, default="Unknown") 
    post_grad_details = Column(String, default="") 
    proof_document_path = Column(String, default="") 
    pdf_document_path = Column(String, default="") 
    
    semesters = relationship("SemesterRecord", back_populates="student", cascade="all, delete-orphan")
    papers = relationship("PaperRecord", back_populates="student", cascade="all, delete-orphan")

class SemesterRecord(Base):
    __tablename__ = "semester_records"
    id = Column(Integer, primary_key=True, index=True)
    registration_no = Column(String, ForeignKey("students.registration_no"))
    semester = Column(String)
    year = Column(String)
    full_marks = Column(String, default="400")
    marks_obtained = Column(String)
    credit = Column(String, default="20")
    sgpa = Column(String)
    student = relationship("Student", back_populates="semesters")

class PaperRecord(Base):
    __tablename__ = "paper_records"
    id = Column(Integer, primary_key=True, index=True)
    registration_no = Column(String, ForeignKey("students.registration_no"))
    course_code = Column(String, default="")
    course_name = Column(String, default="")
    component = Column(String, default="")
    full_marks = Column(String, default="")
    marks_obtained = Column(String, default="")
    credit = Column(String, default="")
    grade = Column(String, default="")
    status = Column(String, default="")
    student = relationship("Student", back_populates="papers")

os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/pdf_repository", exist_ok=True)

STATE_FILE = "upload_state.json"
bg_upload_status = {
    "is_processing": False, "is_paused": False, "pause_requested": False,
    "total_pages": 0, "processed_pages": 0, "extracted_count": 0,
    "status_message": "Idle", "filename": "", "temp_pdf_path": "", 
    "selected_curr": "AUTO", "selected_course": "AUTO", "selected_subject": "AUTO",
    "errors": []
}

def load_upload_state():
    global bg_upload_status
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                bg_upload_status.update(json.load(f))
                if bg_upload_status["is_processing"]:
                    bg_upload_status["is_processing"] = False
                    bg_upload_status["is_paused"] = True
                    bg_upload_status["pause_requested"] = False
                    bg_upload_status["status_message"] = f"Paused at page {bg_upload_status['processed_pages']} (Server Restarted)"
        except: pass

def save_upload_state():
    try:
        with open(STATE_FILE, "w") as f: json.dump(bg_upload_status, f)
    except: pass

def parse_marksheet_with_gemini_vision(page):
    pix = page.get_pixmap(dpi=300)
    image = Image.open(io.BytesIO(pix.tobytes("jpeg")))

    prompt = """
    Extract Calcutta University Grade Sheet data.
    IMPORTANT RULES:
    1. Identify Curriculum: If "CCF" or "MDC" or "Four Year" or "Three Year" is mentioned, return 'CCF'. If it's Choice Based Credit System, return 'CBCS'.
    2. Identify Course (e.g., BA (Hons), BSc (MDC), etc.). Note: "Three Year" usually means MDC or General.
    3. Identify Subject code (e.g., BNGA, BGNM, etc.). If MDC or General, subject may be empty or 'Unknown'.
    
    Return ONLY a valid JSON object matching this exact schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "curriculum": "CBCS",
      "course": "BA (Hons)",
      "subject": "BNGA",
      "passout_year": "2024",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified",
      "papers": [
        {
          "course_code": "CAGM-MDC-3",
          "course_name": "COST ACCOUNTING-II",
          "component": "Theoretical",
          "full_marks": "75",
          "marks_obtained": "40",
          "credit": "3",
          "grade": "B+",
          "status": "P"
        }
      ],
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks_obtained": "248", "credit": "20", "sgpa": "5.624"}
      ]
    }
    """
    
    # 2026 Compatible Google Models
    models_to_try = ['gemini-3.6-flash', 'gemini-3.5-flash', 'gemini-3.0-flash']
    
    last_exception = None
    for model_name in models_to_try:
        for attempt in range(3): 
            try:
                if ai_client_type == "legacy":
                    model = legacy_genai.GenerativeModel(model_name, generation_config={"response_mime_type": "application/json"})
                    response = model.generate_content([prompt, image])
                    txt = response.text.strip()
                elif ai_client_type == "new":
                    response = ai_client.models.generate_content(
                        model=model_name, contents=[prompt, image], 
                        config=types.GenerateContentConfig(response_mime_type="application/json")
                    )
                    txt = response.text.strip()
                else:
                    raise Exception("API Key Missing or SDK Error")

                if txt.startswith("```"):
                    txt = txt.strip("`").strip()
                    if txt.lower().startswith("json"): txt = txt[4:].strip()
                
                return json.loads(txt)
                
            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str: break 
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
                    time.sleep(35)
                    continue
                time.sleep(3)
                continue

    raise last_exception

def process_large_pdf_in_background():
    global bg_upload_status
    db = SessionLocal()
    temp_pdf = bg_upload_status["temp_pdf_path"]
    
    try:
        doc = fitz.open(temp_pdf) if fitz else []
        bg_upload_status["is_processing"] = True
        bg_upload_status["is_paused"] = False
        bg_upload_status["pause_requested"] = False
        save_upload_state()
        
        for page_num in range(bg_upload_status["processed_pages"], len(doc)):
            if bg_upload_status.get("pause_requested"):
                bg_upload_status["is_processing"] = False
                bg_upload_status["is_paused"] = True
                bg_upload_status["status_message"] = f"⏸️ Paused at page {page_num}."
                save_upload_state(); db.close(); doc.close(); return

            try:
                page = doc[page_num]
                
                if ai_client_type:
                    try:
                        if page_num > 0: time.sleep(4.5) 
                        data = parse_marksheet_with_gemini_vision(page)
                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            bg_upload_status["errors"].append(f"Page {page_num+1}: No Reg No detected.")
                            bg_upload_status["processed_pages"] = page_num + 1
                            save_upload_state()
                            continue
                    except Exception as ai_err:
                        ai_error_msg = str(ai_err)
                        if "429" in ai_error_msg.lower() or "exhausted" in ai_error_msg.lower() or "quota" in ai_error_msg.lower():
                            bg_upload_status["is_processing"] = False
                            bg_upload_status["is_paused"] = True
                            bg_upload_status["status_message"] = f"⏸️ AUTO-PAUSED: Google API Quota Exhausted on page {page_num+1}."
                            bg_upload_status["errors"].append(f"Auto-Paused at Page {page_num+1} due to API Limit.")
                            save_upload_state(); db.close(); 
                            if doc: doc.close()
                            return
                        raise Exception(f"AI Parsing Failed: {ai_error_msg}")
                else:
                    raise Exception("API Key Missing or SDK Error")

                # Prepare Normalized Data
                norm_sems, norm_papers = [], []
                for s in data.get("semesters", []):
                    if not s.get("semester"): continue
                    norm_sems.append({
                        "semester": str(s.get("semester")).strip().upper().replace("SEMESTER", "").strip(),
                        "year": str(s.get("year", "")).strip(),
                        "full_marks": str(s.get("full_marks", "400")).strip(),
                        "marks_obtained": str(s.get("marks_obtained", s.get("marks", ""))).strip(),
                        "credit": str(s.get("credit", "20")).strip(),
                        "sgpa": str(s.get("sgpa", "")).strip()
                    })

                for p in data.get("papers", []):
                    if not p.get("course_code") and not p.get("course_name"): continue
                    norm_papers.append({
                        "course_code": str(p.get("course_code", "")).strip(),
                        "course_name": str(p.get("course_name", "")).strip(),
                        "component": str(p.get("component", "")).strip(),
                        "full_marks": str(p.get("full_marks", "")).strip(),
                        "marks_obtained": str(p.get("marks_obtained", "")).strip(),
                        "credit": str(p.get("credit", "")).strip(),
                        "grade": str(p.get("grade", "")).strip(),
                        "status": str(p.get("status", "")).strip()
                    })

                pdf_path = f"uploads/pdf_repository/{reg_no}.pdf"
                try:
                    if fitz:
                        new_pdf = fitz.open()
                        new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                        new_pdf.save(pdf_path)
                        new_pdf.close()
                except Exception: pdf_path = ""

                student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if not student:
                    student = Student(registration_no=reg_no, admission_year="20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown")
                    db.add(student)
                
                # Use dropdown overrides if not AUTO
                scurr = bg_upload_status.get("selected_curr", "AUTO")
                scourse = bg_upload_status.get("selected_course", "AUTO")
                ssubj = bg_upload_status.get("selected_subject", "AUTO")

                student.name = data.get("name", "Unknown")
                student.roll_no = data.get("roll_no", "Unknown")
                student.curriculum = scurr if scurr != "AUTO" and scurr else data.get("curriculum", "Unknown")
                student.course = scourse if scourse != "AUTO" and scourse else data.get("course", "Unknown")
                student.subject = ssubj if ssubj != "AUTO" and ssubj else data.get("subject", "Unknown")
                student.passout_year = str(data.get("passout_year", "Nil")).strip()
                student.overall_cgpa = data.get("overall_cgpa", "N.A.")
                student.overall_grade = data.get("overall_grade", "Fail")
                student.remarks = data.get("remarks", "Qualified")
                if pdf_path: student.pdf_document_path = pdf_path
                
                db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                for s in norm_sems: db.add(SemesterRecord(registration_no=reg_no, **s))
                
                db.query(PaperRecord).filter(PaperRecord.registration_no == reg_no).delete()
                for p in norm_papers: db.add(PaperRecord(registration_no=reg_no, **p))

                bg_upload_status["extracted_count"] += 1
                bg_upload_status["processed_pages"] = page_num + 1
                bg_upload_status["status_message"] = f"Processing page {page_num+1} of {len(doc)} ({bg_upload_status['extracted_count']} saved)..."
                save_upload_state()

                if page_num % 5 == 0: db.commit()

            except Exception as page_err:
                db.rollback()
                bg_upload_status["errors"].append(f"Page {page_num+1} Error: {str(page_err)}")
                bg_upload_status["processed_pages"] = page_num + 1
                save_upload_state()

        if doc: doc.close()
        db.commit()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"✅ DONE! Processed {len(doc)} pages. Saved: {bg_upload_status['extracted_count']}."
        save_upload_state()
        if os.path.exists(temp_pdf): os.remove(temp_pdf)

    except Exception as e:
        db.rollback()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"❌ Fatal Error: {str(e)}"
        save_upload_state()
    finally:
        db.close()

# --- FASTAPI SETUP ---
app = FastAPI(title="Marksheet Portal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.on_event("startup")
def startup_db_setup():
    Base.metadata.create_all(bind=engine)
    try:
        with engine.begin() as conn:
            # 1. Handle SQLite database upgrades
            if "sqlite" in DATABASE_URL:
                tables = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()]
                if "paper_records" not in tables:
                    conn.execute(text("CREATE TABLE paper_records (id INTEGER PRIMARY KEY AUTOINCREMENT, registration_no VARCHAR, course_code VARCHAR, course_name VARCHAR, component VARCHAR, full_marks VARCHAR, marks_obtained VARCHAR, credit VARCHAR, grade VARCHAR, status VARCHAR)"))
                columns = [r[1] for r in conn.execute(text("PRAGMA table_info(students);")).fetchall()]
                if "curriculum" not in columns:
                    conn.execute(text("ALTER TABLE students ADD COLUMN curriculum VARCHAR DEFAULT 'Unknown'"))
            
            # 2. Handle PostgreSQL database upgrades
            elif "postgresql" in DATABASE_URL or "postgres" in DATABASE_URL:
                res = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='students' AND column_name='curriculum';")).fetchone()
                if not res:
                    conn.execute(text("ALTER TABLE students ADD COLUMN curriculum VARCHAR DEFAULT 'Unknown'"))
    except Exception as e:
        print(f"Database migration warning (safe to ignore if DB is up to date): {e}")

    load_upload_state()

@app.get("/")
def serve_frontend(): return FileResponse("index.html")

@app.get("/api/auth/me")
def get_auth_me(user: dict = Depends(get_current_user)): return {"username": user["username"], "role": user["role"]}

@app.get("/api/admin/upload-status")
def get_upload_status(): return bg_upload_status

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(background_tasks: BackgroundTasks, file: UploadFile = File(...), 
                           selected_curr: str = Form("AUTO"), selected_course: str = Form("AUTO"), 
                           selected_subject: str = Form("AUTO"), user: dict = Depends(require_admin)):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()
    
    bg_upload_status.update({
        "is_processing": True, "is_paused": False, "pause_requested": False, "total_pages": total_pages, 
        "processed_pages": 0, "extracted_count": 0, "filename": file.filename, "temp_pdf_path": temp_pdf_path, 
        "selected_curr": selected_curr, "selected_course": selected_course, "selected_subject": selected_subject, "errors": []
    })
    save_upload_state()
    background_tasks.add_task(process_large_pdf_in_background)
    return {"message": "Started"}

@app.post("/api/admin/pause-upload")
def pause_upload(user: dict = Depends(require_admin)):
    if bg_upload_status["is_processing"]: bg_upload_status["pause_requested"] = True; save_upload_state()
    return {"message": "Pausing..."}

@app.post("/api/admin/resume-upload")
def resume_upload(background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    if bg_upload_status["is_paused"]:
        bg_upload_status["is_processing"] = True; bg_upload_status["is_paused"] = False; bg_upload_status["pause_requested"] = False; save_upload_state()
        background_tasks.add_task(process_large_pdf_in_background)
    return {"message": "Resuming..."}

@app.post("/api/admin/cancel-upload")
def cancel_upload(user: dict = Depends(require_admin)):
    global bg_upload_status
    bg_upload_status["is_processing"] = False; bg_upload_status["is_paused"] = False; bg_upload_status["pause_requested"] = False; bg_upload_status["status_message"] = "❌ Process Canceled by User."
    if bg_upload_status.get("temp_pdf_path") and os.path.exists(bg_upload_status["temp_pdf_path"]):
        try: os.remove(bg_upload_status["temp_pdf_path"])
        except: pass
    save_upload_state()
    return {"message": "Canceled"}

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db)):
    student = db.query(Student).options(joinedload(Student.semesters), joinedload(Student.papers)).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Not found")
    return {
        "student": {
            "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
            "admission_year": student.admission_year, "passout_year": student.passout_year, 
            "curriculum": student.curriculum, "course": student.course, "subject": student.subject,
            "cgpa": student.overall_cgpa, "grade": student.overall_grade, "remarks": student.remarks,
            "marksheet_received": student.marksheet_received, "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date, "certificate_issue_date": student.certificate_issue_date,
            "issued_by": student.issued_by, "status": student.post_grad_status, 
            "details": student.post_grad_details, "proof": student.proof_document_path, "pdf_path": student.pdf_document_path
        },
        "semesters": [{"semester": s.semester, "year": s.year, "full_marks": s.full_marks, "marks_obtained": s.marks_obtained, "credit": s.credit, "sgpa": s.sgpa} for s in student.semesters],
        "papers": [{"course_code": p.course_code, "course_name": p.course_name, "component": p.component, "full_marks": p.full_marks, "marks_obtained": p.marks_obtained, "credit": p.credit, "grade": p.grade, "status": p.status} for p in student.papers]
    }

@app.post("/api/admin/update-profile-full/{reg_no}")
async def update_student_profile_full(reg_no: str, payload: dict, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404)
    
    student.name, student.roll_no = payload.get("name"), payload.get("roll_no")
    student.curriculum, student.course, student.subject = payload.get("curriculum"), payload.get("course"), payload.get("subject")
    student.passout_year, student.overall_cgpa, student.overall_grade, student.remarks = payload.get("passout_year"), payload.get("cgpa"), payload.get("grade"), payload.get("remarks")
    
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    for sem in payload.get("semesters", []): 
        db.add(SemesterRecord(registration_no=reg_no, semester=sem.get("semester"), year=sem.get("year"), full_marks=sem.get("full_marks"), marks_obtained=sem.get("marks_obtained"), credit=sem.get("credit"), sgpa=sem.get("sgpa")))
    db.query(PaperRecord).filter(PaperRecord.registration_no == reg_no).delete()
    for p in payload.get("papers", []): 
        db.add(PaperRecord(registration_no=reg_no, **p))
    db.commit()
    return {"message": "Success"}

@app.post("/api/admin/update-issuance-detailed/{reg_no}")
async def update_issuance_detailed(
    reg_no: str, marksheet_received: bool = Form(...), certificate_received: bool = Form(...),
    marksheet_issue_date: str = Form(""), certificate_issue_date: str = Form(""),
    issued_by: str = Form(""), db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")
    student.marksheet_received, student.certificate_received = marksheet_received, certificate_received
    student.marksheet_issue_date, student.certificate_issue_date, student.issued_by = marksheet_issue_date, certificate_issue_date, issued_by
    db.commit()
    return {"message": "Updated!"}

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(
    reg_no: str, marksheet_received: bool = Form(...), certificate_received: bool = Form(...),
    status: str = Form(...), details: str = Form(""), proof_file: UploadFile = File(None), db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    student.marksheet_received, student.certificate_received = marksheet_received, certificate_received
    student.post_grad_status, student.post_grad_details = status, details
    if proof_file:
        safe_filename = f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}"
        file_path = os.path.join("uploads", safe_filename)
        with open(file_path, "wb") as buffer: shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path
    db.commit()
    return {"message": "Record updated!"}

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db)): return db.query(Student).options(joinedload(Student.semesters)).all()

@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if student.pdf_document_path and os.path.exists(student.pdf_document_path): os.remove(student.pdf_document_path)
    db.delete(student)
    db.commit()
    return {"message": "Deleted"}

@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    db.query(PaperRecord).delete(); db.query(SemesterRecord).delete(); db.query(Student).delete(); db.commit()
    shutil.rmtree("uploads/pdf_repository", ignore_errors=True); os.makedirs("uploads/pdf_repository", exist_ok=True)
    return {"message": "Cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

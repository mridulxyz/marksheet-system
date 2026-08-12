import os
import re
import shutil
import secrets
import json
import base64
import time
import gc
import io
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# Safe Imports
try:
    import pymupdf as fitz  # PyMuPDF
except ImportError:
    try:
        import fitz # Fallback for older versions
    except ImportError:
        fitz = None
    
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    print("WARNING: 'google-genai' package is not installed! Gemini OCR will be completely skipped.")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base, joinedload

# --- SECURITY (MULTI-USER AUTHENTICATION) ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")
USER1_USERNAME = os.getenv("USER1_USERNAME", "staff1")
USER1_PASSWORD = os.getenv("USER1_PASSWORD", "staff123")
USER2_USERNAME = os.getenv("USER2_USERNAME", "staff2")
USER2_PASSWORD = os.getenv("USER2_PASSWORD", "staff123")

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    if secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(password, ADMIN_PASSWORD):
        return {"username": username, "role": "admin"}
    if secrets.compare_digest(username, USER1_USERNAME) and secrets.compare_digest(password, USER1_PASSWORD):
        return {"username": username, "role": "user"}
    if secrets.compare_digest(username, USER2_USERNAME) and secrets.compare_digest(password, USER2_PASSWORD):
        return {"username": username, "role": "user"}
    raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Basic"})

def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin permissions required.")
    return user

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if (genai and GEMINI_API_KEY) else None

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, pool_pre_ping=True, pool_recycle=300,        
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
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
    course = Column(String, default="Unknown Course")
    subject = Column(String, default="BNGA")
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

# NEW: Detailed Paper Record Model for full digitization
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

# --- GLOBAL BACKGROUND PROGRESS & PERSISTENCE ---
STATE_FILE = "upload_state.json"
bg_upload_status = {
    "is_processing": False, "is_paused": False, "pause_requested": False,
    "total_pages": 0, "processed_pages": 0, "extracted_count": 0,
    "status_message": "Idle", "filename": "", "temp_pdf_path": "", "selected_course": "AUTO"
}

def load_upload_state():
    global bg_upload_status
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved_state = json.load(f)
                bg_upload_status.update(saved_state)
                # If server crashed while processing, set to paused so user can resume
                if bg_upload_status["is_processing"]:
                    bg_upload_status["is_processing"] = False
                    bg_upload_status["is_paused"] = True
                    bg_upload_status["pause_requested"] = False
                    bg_upload_status["status_message"] = f"Paused at page {bg_upload_status['processed_pages']} (Server Restarted)"
        except Exception: pass

def save_upload_state():
    global bg_upload_status
    try:
        with open(STATE_FILE, "w") as f: json.dump(bg_upload_status, f)
    except Exception: pass

def auto_repair_passout_year(student_obj):
    py = str(student_obj.passout_year).strip()
    is_fail = ("fail" in str(student_obj.overall_grade).lower() or "not cleared" in str(student_obj.remarks).lower() or student_obj.overall_grade in ["", "None", "N.A."])
    
    if is_fail:
        if py.lower() != "nil":
            student_obj.passout_year = "Nil"
            return True
    else:
        if py.lower() in ["nil", "none", "unknown", "", "n.a.", "null"]:
            max_yr = None
            max_val = 0
            for sem in student_obj.semesters:
                if sem.year:
                    m = re.search(r'(20[1-3]\d)', str(sem.year))
                    if m:
                        val = int(m.group(1))
                        # Support Sem 8
                        if sem.semester in ["VI", "6", "VIII", "8"]:
                            max_yr = str(val)
                            break
                        if val > max_val:
                            max_val = val
                            max_yr = str(val)
            if max_yr:
                student_obj.passout_year = max_yr
                return True
    return False

def parse_marksheet_with_gemini_vision(page):
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("jpeg")
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """
    You are an expert transcript parser. Carefully inspect this Calcutta University Grade Sheet image (which could span up to 8 semesters).
    Return ONLY a valid JSON object matching this exact schema covering BOTH the detailed paper breakdown AND the semester summary:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination",
      "subject": "BNGA",
      "passout_year": "2024",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified with Honours",
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
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"},
        {"semester": "VIII", "year": "2024", "full_marks": "400", "marks": "310", "credit": "20", "sgpa": "7.705"}
      ]
    }

    RULES:
    1. Extract ALL rows from the detailed subjects table into "papers". Include components like Theoretical/Tutorial/Practical. Ignore 'Total' rows if they merge components, just extract the base marks if possible.
    2. Extract ALL available semesters (I to VIII) from the summary table into "semesters".
    3. OVERALL GRADE/CGPA: Only populate if a final Cumulative CGPA/Letter Grade is visible on the sheet. Otherwise "N.A."
    4. If 'Semester not cleared' is in Remarks, set 'overall_cgpa': 'N.A.' and 'overall_grade': 'Fail / Semester Not Cleared'.
    """

    models_to_try = ['gemini-1.5-flash', 'gemini-3.5-flash', 'gemini-1.5-pro']
    
    last_exception = None
    for model_name in models_to_try:
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = ai_client.models.generate_content(
                    model=model_name, contents=[prompt, image],
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                result_json = response.text.strip()
                if result_json.startswith("```"):
                    result_json = result_json.strip("`").strip()
                    if result_json.lower().startswith("json"):
                        result_json = result_json[4:].strip()
                return json.loads(result_json)
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if "404" in err_msg or "not found" in err_msg: break 
                if ("429" in err_msg or "rate" in err_msg): time.sleep(3)
                else: break 
    raise last_exception

def extract_reg_no_bulletproof(text: str):
    if not text: return None, "Text empty"
    t = re.sub(r'[—–_~]', '-', text)
    t_fixed = t.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    match = re.search(r'(?:Registration|Regn|Reg|Registra)[^\d]{0,15}([0-9\-\s\.\/]{13,22})', t_fixed, re.IGNORECASE)
    if match:
        digits = re.sub(r'\D', '', match.group(1))
        if len(digits) >= 13:
            d = digits[:13]
            return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}", "Reg Label"
    all_digits_clusters = re.findall(r'\b\d{13}\b', re.sub(r'\D', ' ', t_fixed))
    if all_digits_clusters:
        d = all_digits_clusters[0]
        return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}", "13-Digit Cluster"
    return None, "No 13-digit pattern found"

def process_large_pdf_in_background():
    global bg_upload_status
    db = SessionLocal()
    temp_pdf_path = bg_upload_status["temp_pdf_path"]
    selected_course = bg_upload_status["selected_course"]
    
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        total_pages = len(doc)
        
        bg_upload_status["is_processing"] = True
        bg_upload_status["is_paused"] = False
        bg_upload_status["pause_requested"] = False
        bg_upload_status["total_pages"] = total_pages
        save_upload_state()
        ai_quota_exceeded = False
        
        for page_num in range(bg_upload_status["processed_pages"], total_pages):
            if bg_upload_status.get("pause_requested"):
                bg_upload_status["is_processing"] = False
                bg_upload_status["is_paused"] = True
                bg_upload_status["status_message"] = f"⏸️ Paused at page {page_num} of {total_pages}. Click Resume to continue."
                save_upload_state()
                db.close()
                if doc: doc.close()
                return

            try:
                page = doc[page_num]
                reg_no = None
                normalized_semesters = []
                normalized_papers = []
                remarks = "Qualified"
                subject = "BNGA"
                passout_year = "Nil"
                
                # AI Parsing
                if ai_client and not ai_quota_exceeded:
                    try:
                        if page_num > 0: time.sleep(1.5) 
                        data = parse_marksheet_with_gemini_vision(page)
                        reg_no = data.get("registration_no")
                        
                        if not reg_no or reg_no == "null":
                            reg_no = None
                        else:
                            roll_no = data.get("roll_no", "Unknown")
                            name = data.get("name", "Unknown Student")
                            course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "Unknown")
                            subject = data.get("subject", "BNGA")
                            passout_year = str(data.get("passout_year", "Nil")).strip()
                            remarks = data.get("remarks", "Qualified")
                            overall_cgpa = data.get("overall_cgpa", "N.A.")
                            overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")

                            if "not cleared" in str(remarks).lower() or overall_grade.lower() in ["none", "null", "n.a.", "", "fail / semester not cleared"]:
                                overall_cgpa = "N.A."
                                overall_grade = "Fail / Semester Not Cleared"
                            
                            for sem in data.get("semesters", []):
                                if not isinstance(sem, dict): continue
                                raw_s = str(sem.get("semester", "")).strip().upper()
                                raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                                if not raw_s: continue
                                normalized_semesters.append({
                                    "semester": raw_s, "year": str(sem.get("year", "")).strip(),
                                    "full_marks": str(sem.get("full_marks", "400")).strip(),
                                    "marks": str(sem.get("marks", "")).strip(),
                                    "credit": str(sem.get("credit", "20")).strip(),
                                    "sgpa": str(sem.get("sgpa", "")).strip()
                                })

                            for p in data.get("papers", []):
                                if not isinstance(p, dict): continue
                                normalized_papers.append({
                                    "course_code": str(p.get("course_code", "")).strip(),
                                    "course_name": str(p.get("course_name", "")).strip(),
                                    "component": str(p.get("component", "")).strip(),
                                    "full_marks": str(p.get("full_marks", "")).strip(),
                                    "marks_obtained": str(p.get("marks_obtained", "")).strip(),
                                    "credit": str(p.get("credit", "")).strip(),
                                    "grade": str(p.get("grade", "")).strip(),
                                    "status": str(p.get("status", "")).strip()
                                })
                    except Exception as ai_err:
                        print(f"[Page {page_num+1}] AI Error: {str(ai_err)}")
                        if "429" in str(ai_err).lower() or "rate" in str(ai_err).lower(): ai_quota_exceeded = True

                # Fallback if AI fails completely
                if not ai_client or ai_quota_exceeded or not reg_no:
                    full_text = page.get_text("text") or ""
                    reg_no_found, _ = extract_reg_no_bulletproof(full_text)
                    if reg_no_found: reg_no = reg_no_found
                    if not reg_no:
                        bg_upload_status["processed_pages"] = page_num + 1
                        continue
                    roll_no = "Unknown"
                    name = "Unknown Student"
                    course = selected_course
                    overall_cgpa = "N.A."
                    overall_grade = "Fail"

                # DB Insert / Update
                pdf_repo_path = f"uploads/pdf_repository/{reg_no}.pdf"
                try:
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    new_pdf.save(pdf_repo_path)
                    new_pdf.close()
                except Exception: pdf_repo_path = ""

                existing = db.query(Student).filter(Student.registration_no == reg_no).first()
                if not existing:
                    admission_year = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    existing = Student(registration_no=reg_no, admission_year=admission_year)
                    db.add(existing)
                
                existing.name = name
                existing.roll_no = roll_no
                existing.course = course
                existing.subject = subject
                existing.passout_year = passout_year
                existing.overall_cgpa = overall_cgpa
                existing.overall_grade = overall_grade
                existing.remarks = remarks
                if pdf_repo_path: existing.pdf_document_path = pdf_repo_path
                
                # Delete existing related records and insert fresh to avoid duplicates in deep structures
                db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                for sem in normalized_semesters:
                    db.add(SemesterRecord(registration_no=reg_no, **sem))
                
                db.query(PaperRecord).filter(PaperRecord.registration_no == reg_no).delete()
                for p in normalized_papers:
                    db.add(PaperRecord(registration_no=reg_no, **p))

                bg_upload_status["extracted_count"] += 1
                bg_upload_status["processed_pages"] = page_num + 1
                bg_upload_status["status_message"] = f"Processing page {page_num+1} of {total_pages} ({bg_upload_status['extracted_count']} records saved)..."
                save_upload_state()

                if page_num % 5 == 0:
                    db.commit()
                    gc.collect()

            except Exception as page_err:
                print(f"Error on page {page_num}: {page_err}")
                db.rollback()
                bg_upload_status["processed_pages"] = page_num + 1
                save_upload_state()
                continue

        if doc: doc.close()
        db.commit()
        
        bg_upload_status["is_processing"] = False
        bg_upload_status["is_paused"] = False
        bg_upload_status["status_message"] = f"🎉 ✅ EXTRACTION DONE! Processed {total_pages} pages."
        save_upload_state()
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

    except Exception as e:
        db.rollback()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"❌ Error: {str(e)}"
        save_upload_state()
    finally:
        db.close()

# --- FASTAPI APP SETUP ---
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
    # Ensure paper_records exists if sqlite didn't generate it natively on old DB
    with engine.begin() as conn:
        if "sqlite" in DATABASE_URL:
            tables = [r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';")).fetchall()]
            if "paper_records" not in tables:
                conn.execute(text("""CREATE TABLE paper_records (
                    id INTEGER PRIMARY KEY, registration_no VARCHAR, course_code VARCHAR, course_name VARCHAR,
                    component VARCHAR, full_marks VARCHAR, marks_obtained VARCHAR, credit VARCHAR, grade VARCHAR, status VARCHAR
                )"""))
    load_upload_state()

@app.get("/")
def serve_frontend(): return FileResponse("index.html")

@app.get("/api/auth/me")
def get_auth_me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

@app.get("/api/admin/upload-status")
def get_upload_status(user: dict = Depends(get_current_user)):
    return bg_upload_status

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(background_tasks: BackgroundTasks, file: UploadFile = File(...), selected_course: str = Form("AUTO"), user: dict = Depends(require_admin)):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()

    bg_upload_status.update({
        "is_processing": True, "is_paused": False, "pause_requested": False,
        "total_pages": total_pages, "processed_pages": 0, "extracted_count": 0,
        "filename": file.filename, "temp_pdf_path": temp_pdf_path, "selected_course": selected_course
    })
    save_upload_state()
    background_tasks.add_task(process_large_pdf_in_background)
    return {"message": f"🚀 Started processing {total_pages} page(s)."}

@app.post("/api/admin/pause-upload")
def pause_upload(user: dict = Depends(require_admin)):
    if bg_upload_status["is_processing"]:
        bg_upload_status["pause_requested"] = True
        save_upload_state()
        return {"message": "Pause requested."}
    return {"message": "No active upload to pause."}

@app.post("/api/admin/resume-upload")
def resume_upload(background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    if not bg_upload_status["is_paused"]: raise HTTPException(400, "No paused upload found.")
    bg_upload_status["is_processing"] = True
    bg_upload_status["is_paused"] = False
    bg_upload_status["pause_requested"] = False
    save_upload_state()
    background_tasks.add_task(process_large_pdf_in_background)
    return {"message": "Resuming upload..."}

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).options(joinedload(Student.semesters), joinedload(Student.papers)).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student record not found.")
    if auto_repair_passout_year(student): db.commit()

    return {
        "student": {
            "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
            "admission_year": student.admission_year, "passout_year": student.passout_year, 
            "course": student.course, "subject": student.subject,
            "cgpa": student.overall_cgpa, "grade": student.overall_grade,
            "remarks": student.remarks,
            "marksheet_received": student.marksheet_received, "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date, "certificate_issue_date": student.certificate_issue_date,
            "issued_by": student.issued_by, "status": student.post_grad_status, 
            "details": student.post_grad_details, "proof": student.proof_document_path, "pdf_path": student.pdf_document_path
        },
        "semesters": [{"semester": s.semester, "year": s.year, "full_marks": s.full_marks, "marks": s.marks_obtained, "credit": s.credit, "sgpa": s.sgpa} for s in student.semesters],
        "papers": [{"course_code": p.course_code, "course_name": p.course_name, "component": p.component, "full_marks": p.full_marks, "marks_obtained": p.marks_obtained, "credit": p.credit, "grade": p.grade, "status": p.status} for p in student.papers]
    }

@app.post("/api/admin/update-profile-full/{reg_no}")
async def update_student_profile_full(reg_no: str, payload: dict, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")

    student.name = payload.get("name", student.name)
    student.roll_no = payload.get("roll_no", student.roll_no)
    student.course = payload.get("course", student.course)
    student.subject = payload.get("subject", student.subject)
    student.passout_year = payload.get("passout_year", student.passout_year)
    student.overall_cgpa = payload.get("cgpa", student.overall_cgpa)
    student.overall_grade = payload.get("grade", student.overall_grade)
    student.remarks = payload.get("remarks", student.remarks)

    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    for sem in payload.get("semesters", []):
        db.add(SemesterRecord(registration_no=reg_no, **sem))
        
    db.query(PaperRecord).filter(PaperRecord.registration_no == reg_no).delete()
    for p in payload.get("papers", []):
        db.add(PaperRecord(registration_no=reg_no, **p))

    db.commit()
    return {"message": "Profile updated successfully!"}

@app.post("/api/admin/update-issuance-detailed/{reg_no}")
async def update_issuance_detailed(
    reg_no: str, marksheet_received: bool = Form(...), certificate_received: bool = Form(...),
    marksheet_issue_date: str = Form(""), certificate_issue_date: str = Form(""),
    issued_by: str = Form(""), db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")
    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.marksheet_issue_date = marksheet_issue_date
    student.certificate_issue_date = certificate_issue_date
    student.issued_by = issued_by
    db.commit()
    return {"message": "Issuance details updated successfully!"}

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(
    reg_no: str, course: str = Form(...), marksheet_received: bool = Form(...), certificate_received: bool = Form(...),
    status: str = Form(...), details: str = Form(""), proof_file: UploadFile = File(None),
    db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    student.course = course
    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.post_grad_status = status
    student.post_grad_details = details
    if proof_file:
        safe_filename = f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}"
        file_path = os.path.join("uploads", safe_filename)
        with open(file_path, "wb") as buffer: shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path
    db.commit()
    return {"message": "Career Record updated!"}

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    return db.query(Student).options(joinedload(Student.semesters)).all()

@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if student.pdf_document_path and os.path.exists(student.pdf_document_path): os.remove(student.pdf_document_path)
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    db.query(PaperRecord).filter(PaperRecord.registration_no == reg_no).delete()
    db.delete(student)
    db.commit()
    return {"message": f"Student {reg_no} deleted successfully"}

@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    db.query(PaperRecord).delete()
    db.query(SemesterRecord).delete()
    db.query(Student).delete()
    db.commit()
    try:
        shutil.rmtree("uploads/pdf_repository")
        os.makedirs("uploads/pdf_repository", exist_ok=True)
    except: pass
    return {"message": "All student records cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

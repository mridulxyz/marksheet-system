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

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, text
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
        raise HTTPException(status_code=403, detail="Admin permissions required to perform this operation.")
    return user

# --- GEMINI CLIENT SETUP ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ai_client = genai.Client(api_key=GEMINI_API_KEY) if (genai and GEMINI_API_KEY) else None

# --- DATABASE SETUP ---
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

os.makedirs("uploads", exist_ok=True)
os.makedirs("uploads/pdf_repository", exist_ok=True)

# --- GLOBAL BACKGROUND PROGRESS & PERSISTENCE ---
STATE_FILE = "upload_state.json"
bg_upload_status = {
    "is_processing": False,
    "is_paused": False,
    "pause_requested": False,
    "total_pages": 0,
    "processed_pages": 0,
    "extracted_count": 0,
    "status_message": "Idle",
    "filename": "",
    "temp_pdf_path": "",
    "selected_course": "AUTO"
}

def load_upload_state():
    global bg_upload_status
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved_state = json.load(f)
                bg_upload_status.update(saved_state)
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

# --- HELPER: AUTO-REPAIR PASSOUT YEAR ---
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
                        if sem.semester in ["VIII", "8", "VIII.", "8th", "VI", "6", "VI.", "6th"]:
                            max_yr = str(val)
                            break
                        if val > max_val:
                            max_val = val
                            max_yr = str(val)
            if max_yr:
                student_obj.passout_year = max_yr
                return True
    return False

# --- GEMINI VISION PARSER ---
def parse_marksheet_with_gemini_vision(page):
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("jpeg")
    image = Image.open(io.BytesIO(img_bytes))

    prompt = """
    You are an expert transcript parser. Carefully inspect this Calcutta University Grade Sheet image. 
    It could be a single-semester Grade Card (e.g., Sem I or III) OR a multi-semester summary up to Semester VIII.
    Return ONLY a valid JSON object matching this exact schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)",
      "subject": "BNGA",
      "passout_year": "2024",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified with Honours",
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"},
        {"semester": "III", "year": "2025", "full_marks": "525", "marks": "329", "credit": "21", "sgpa": "6.267"}
      ]
    }

    VERIFICATION RULES FOR 100% ACCURACY:
    1. Extract ALL available semesters (I, II, III, IV, V, VI, VII, VIII) shown into the `semesters` array. Do not invent missing ones.
    2. For Single-Semester Grade Cards, extract only that semester. Set overall_cgpa and overall_grade to "N.A." if not present, and map the SGPA strictly to the specific semester.
    3. OVERALL GRADE/CGPA: Only populate if a final Cumulative CGPA/Letter Grade is visible.
    4. FAILED/NOT CLEARED: If 'Semester not cleared' is in Remarks, set 'overall_cgpa': 'N.A.' and 'overall_grade': 'Fail / Semester Not Cleared'.
    """

    models_to_try = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-3.5-flash', 'gemini-3.5-flash-lite']
    
    last_exception = None
    for model_name in models_to_try:
        for attempt in range(2):
            try:
                response = ai_client.models.generate_content(
                    model=model_name, contents=[prompt, image],
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
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
                if "429" in err_msg or "quota" in err_msg: time.sleep(3)
                else: break 
    raise last_exception

# --- LOCAL FALLBACK PARSER ---
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

def parse_summary_table_local(full_text: str):
    sems = []
    cgpa = "N.A."
    grade = "N.A."
    remarks = "Qualified"
    if not full_text: return sems, cgpa, grade, remarks

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', full_text)
    is_not_cleared = bool(re.search(r'(?:Semester\s*not\s*cleared|not\s*cleared)', text_clean, re.IGNORECASE))
    rem_match = re.search(r'Remarks\s*[\:\.]*\s*([^\n\r]+)', text_clean, re.IGNORECASE)
    if rem_match: remarks = rem_match.group(1).strip()

    grade_m = re.search(r'CGPA.*?([A-Z\+]{1,3})', text_clean)
    if grade_m: grade = grade_m.group(1)
    
    if is_not_cleared:
        grade = "Fail / Semester Not Cleared"

    lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
    sem_keys = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII']

    for line in lines:
        for sem in sem_keys:
            if re.search(rf'\b{sem}\b', line):
                years = re.findall(r'\b(20\d\d)\b', line)
                integers = [i for i in re.findall(r'\b(\d{2,3})\b', line) if i not in years]
                floats = re.findall(r'\b([1-9]\.\d{2,3})\b', line)
                if years and integers and floats:
                    if not any(item["semester"] == sem for item in sems):
                        sems.append({
                            "semester": sem, "year": years[0],
                            "full_marks": integers[0] if len(integers) >= 1 else "400",
                            "marks": integers[1] if len(integers) >= 2 else (integers[0] if integers else "-"),
                            "credit": integers[2] if len(integers) >= 3 else "20",
                            "sgpa": floats[0]
                        })

    sem_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8}
    sems = sorted(sems, key=lambda x: sem_order.get(x["semester"], 99))
    all_floats = re.findall(r'\b([1-9]\.\d{2,3})\b', text_clean)
    valid_gpas = [f for f in all_floats if 1.0 <= float(f) <= 10.0]
    if valid_gpas: cgpa = valid_gpas[-1]

    return sems, cgpa, grade, remarks

# --- BACKGROUND WORKER FOR PDFs ---
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
            # Check for Pause Signal
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
                remarks = "Qualified"
                subject = "Unknown"
                passout_year = "Nil"
                
                # --- STRATEGY 1: GEMINI VISION ---
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
                            subject = data.get("subject", "Unknown")
                            passout_year = str(data.get("passout_year", "Nil")).strip()
                            remarks = data.get("remarks", "Qualified")
                            overall_cgpa = data.get("overall_cgpa", "N.A.")
                            overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")

                            if "not cleared" in str(remarks).lower() or overall_grade.lower() in ["none", "null", "", "fail / semester not cleared"]:
                                overall_cgpa = "N.A."
                                overall_grade = "Fail / Semester Not Cleared"
                            
                            raw_semesters = data.get("semesters", [])
                            for sem in raw_semesters:
                                if not isinstance(sem, dict): continue
                                raw_s = str(sem.get("semester") or sem.get("sem") or "").strip().upper()
                                raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                                if not raw_s: continue
                                yr_val = str(sem.get("year") or sem.get("exam_year") or "").strip()
                                sgpa_val = str(sem.get("sgpa") or sem.get("gpa") or "").strip()
                                marks_val = str(sem.get("marks") or sem.get("marks_obtained") or "").strip()
                                if (not yr_val or yr_val in ["-", "N.A."]) and (not sgpa_val or sgpa_val in ["-", "N.A."]):
                                    continue
                                normalized_semesters.append({
                                    "semester": raw_s, "year": yr_val if yr_val not in ["-", "N.A."] else "-",
                                    "full_marks": str(sem.get("full_marks") or "400").strip(),
                                    "marks": marks_val if marks_val not in ["-", "N.A."] else "-",
                                    "credit": str(sem.get("credit") or "20").strip(),
                                    "sgpa": sgpa_val if sgpa_val not in ["-", "N.A."] else "N.A."
                                })
                    except Exception as ai_err:
                        print(f"[Page {page_num+1}] Gemini API Issue: {str(ai_err)}")
                        if "429" in str(ai_err).lower() or "rate" in str(ai_err).lower(): ai_quota_exceeded = True

                # --- STRATEGY 2: LOCAL FALLBACK ---
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
                    normalized_semesters, overall_cgpa, overall_grade, remarks = parse_summary_table_local(full_text)
                
                # --- STRICT ENFORCEMENT & BULLETPROOF FALLBACK ---
                if passout_year.lower() in ["nil", "unknown", "", "none", "null", "n.a."]:
                    max_yr = None
                    max_val = 0
                    for sem in normalized_semesters:
                        m = re.search(r'(20[1-3]\d)', str(sem["year"]))
                        if m:
                            val = int(m.group(1))
                            if sem["semester"] in ["VIII", "8", "VI", "6"]:
                                max_yr = str(val)
                                break
                            if val > max_val:
                                max_val = val
                                max_yr = str(val)
                    if max_yr: passout_year = max_yr

                # Save PDF copy
                pdf_repo_path = f"uploads/pdf_repository/{reg_no}.pdf"
                try:
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    new_pdf.save(pdf_repo_path)
                    new_pdf.close()
                except Exception:
                    pdf_repo_path = ""

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.subject = subject
                    existing_student.passout_year = passout_year
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    existing_student.remarks = remarks
                    if pdf_repo_path: existing_student.pdf_document_path = pdf_repo_path
                    
                    # Instead of deleting all, we UPSERT individual semesters so we don't lose previous data
                    for sem in normalized_semesters:
                        existing_sem = db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no, SemesterRecord.semester == sem["semester"]).first()
                        if existing_sem:
                            existing_sem.year = sem["year"]
                            existing_sem.full_marks = sem["full_marks"]
                            existing_sem.marks_obtained = sem["marks"]
                            existing_sem.credit = sem["credit"]
                            existing_sem.sgpa = sem["sgpa"]
                        else:
                            db.add(SemesterRecord(registration_no=reg_no, semester=sem["semester"], year=sem["year"], full_marks=sem["full_marks"], marks_obtained=sem["marks"], credit=sem["credit"], sgpa=sem["sgpa"]))
                else:
                    admission_year = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    student = Student(
                        registration_no=reg_no, roll_no=roll_no, name=name, admission_year=admission_year, 
                        passout_year=passout_year, course=course, subject=subject, overall_cgpa=overall_cgpa, 
                        overall_grade=overall_grade, remarks=remarks, pdf_document_path=pdf_repo_path
                    )
                    db.add(student)
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(registration_no=reg_no, semester=sem["semester"], year=sem["year"], full_marks=sem["full_marks"], marks_obtained=sem["marks"], credit=sem["credit"], sgpa=sem["sgpa"]))

                bg_upload_status["extracted_count"] += 1
                bg_upload_status["processed_pages"] = page_num + 1
                bg_upload_status["status_message"] = f"Processing page {page_num+1} of {total_pages} ({bg_upload_status['extracted_count']} records saved)..."
                save_upload_state()

                if page_num % 5 == 0:
                    db.commit()
                    gc.collect()

            except Exception as page_err:
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
        
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    except Exception as e:
        db.rollback()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"❌ Error: {str(e)}"
        save_upload_state()
    finally:
        db.close()

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Shyampur Siddheswari Mahavidyalaya Marksheet Portal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.on_event("startup")
def startup_db_setup():
    Base.metadata.create_all(bind=engine)
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
        return {"message": "Pause requested. Process will halt shortly."}
    return {"message": "No active upload to pause."}

@app.post("/api/admin/resume-upload")
def resume_upload(background_tasks: BackgroundTasks, user: dict = Depends(require_admin)):
    if not bg_upload_status["is_paused"] or not os.path.exists(bg_upload_status["temp_pdf_path"]):
        raise HTTPException(400, "No paused upload found or file deleted.")
    
    bg_upload_status["is_processing"] = True
    bg_upload_status["is_paused"] = False
    bg_upload_status["pause_requested"] = False
    save_upload_state()
    background_tasks.add_task(process_large_pdf_in_background)
    return {"message": "Resuming upload process..."}

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).options(joinedload(Student.semesters)).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student record not found.")
    if auto_repair_passout_year(student): db.commit()

    return {
        "student": {
            "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
            "admission_year": student.admission_year, "passout_year": student.passout_year, 
            "course": student.course, "subject": student.subject or "Unknown",
            "cgpa": student.overall_cgpa, "grade": student.overall_grade,
            "remarks": student.remarks or "Qualified",
            "marksheet_received": student.marksheet_received, "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date, "certificate_issue_date": student.certificate_issue_date,
            "issued_by": student.issued_by, "status": student.post_grad_status, 
            "details": student.post_grad_details, "proof": student.proof_document_path, "pdf_path": student.pdf_document_path
        },
        "semesters": [{"semester": s.semester, "year": s.year, "full_marks": s.full_marks, "marks": s.marks_obtained, "credit": s.credit, "sgpa": s.sgpa} for s in student.semesters]
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

    new_semesters = payload.get("semesters", [])
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    for sem in new_semesters:
        db.add(SemesterRecord(registration_no=reg_no, semester=sem.get("semester"), year=sem.get("year"), full_marks=sem.get("full_marks", "400"), marks_obtained=sem.get("marks"), credit=sem.get("credit", "20"), sgpa=sem.get("sgpa")))
    
    db.commit()
    return {"message": "Profile updated successfully!"}

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    return db.query(Student).options(joinedload(Student.semesters)).all()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

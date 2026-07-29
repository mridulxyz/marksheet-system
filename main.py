import os
import re
import shutil
import secrets
import json
import base64
import time
from PIL import Image

# Safe Imports
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import openai
except ImportError:
    openai = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base

# --- SECURITY (ADMIN LOGIN) ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- OPENAI CLIENT SETUP ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if (openai and OPENAI_API_KEY) else None

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,      # Auto-reconnects dropped SSL handles
    pool_recycle=300,        # Recycles connections every 5 mins
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
    course = Column(String, default="Unknown Course")
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    post_grad_status = Column(String, default="Unemployed")
    post_grad_details = Column(String, default="") 
    proof_document_path = Column(String, default="") 
    
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

# --- OPENAI GPT-4o-MINI VISION PARSER (WITH AUTO-RETRY RATE LIMITER) ---

def parse_marksheet_with_openai_vision(page):
    pix = page.get_pixmap(dpi=130)
    img_bytes = pix.tobytes("jpeg")
    base64_image = base64.b64encode(img_bytes).decode('utf-8')

    prompt = """
    You are an expert transcript parser for Calcutta University Grade Sheets. Analyze the provided grade sheet image and return ONLY a valid JSON object in this exact schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"},
        {"semester": "II", "year": "2020", "full_marks": "400", "marks": "310", "credit": "20", "sgpa": "7.705"},
        {"semester": "III", "year": "2022", "full_marks": "500", "marks": "330", "credit": "26", "sgpa": "6.899"},
        {"semester": "IV", "year": "2023", "full_marks": "500", "marks": "372", "credit": "26", "sgpa": "7.367"},
        {"semester": "V", "year": "2023", "full_marks": "400", "marks": "263", "credit": "24", "sgpa": "6.527"},
        {"semester": "VI", "year": "2024", "full_marks": "400", "marks": "270", "credit": "24", "sgpa": "6.686"}
      ]
    }

    STRICT CRITICAL RULES:
    1. OVERALL GRADE: Look ONLY at row 'VI' in the bottom summary table under column 'Letter Grade' (e.g. B+, A+, A, B, C+, C, D, O). Do NOT read status 'P' (Passed) or subject grades from top subject tables.
    2. OVERALL CGPA: Look ONLY at row 'VI' in the bottom summary table under column 'CGPA' (e.g. 6.819).
    3. SEMESTERS: You MUST extract ALL cleared semester rows (I, II, III, IV, V, VI) from the bottom summary table. Each item in 'semesters' MUST contain: 'semester' (I, II, III, IV, V, VI), 'year', 'full_marks', 'marks', 'credit', 'sgpa'.
    4. IF SEMESTER NOT CLEARED: If 'Semester not cleared' is printed in Remarks, set 'overall_cgpa': 'N.A.' and 'overall_grade': 'Fail / Semester Not Cleared'.
    5. COURSE TITLE: Omit 'Semester - VI' from course title. Use 'B.A. (Honours) Examination (Under CBCS)'.
    """

    # Auto-retry up to 3 times on 1-minute rate limit spikes
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            result_json = response.choices[0].message.content
            return json.loads(result_json)
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                # Sleep 4s on first 429, 8s on second 429 to let rate-limit window reset
                time.sleep((attempt + 1) * 4)
            else:
                raise e

# --- LOCAL FALLBACK PARSER ---

def extract_text_rows_from_rect(page, rect_box):
    try:
        words = page.get_text("words", clip=rect_box)
        if not words: return ""
        sorted_words = sorted(words, key=lambda w: (round(w[1] / 5.0), w[0]))

        lines = []
        current_line = []
        last_y = None

        for w in sorted_words:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            if last_y is None or abs(y0 - last_y) <= 5:
                current_line.append(word)
                last_y = y0
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
                last_y = y0

        if current_line: lines.append(" ".join(current_line))
        return "\n".join(lines)
    except Exception:
        return page.get_text("text", clip=rect_box) or ""

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

def extract_roll_no_bulletproof(text: str):
    if not text: return "Unknown"
    t = re.sub(r'[—–_~]', '-', text)
    t_fixed = t.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1').replace('S', '5')

    all_digits_clusters = re.findall(r'\b\d{12}\b', re.sub(r'\D', ' ', t_fixed))
    if all_digits_clusters:
        d = all_digits_clusters[0]
        return f"{d[:6]}-{d[6:8]}-{d[8:]}"

    return "Unknown"

def extract_name_bulletproof(text: str):
    if not text: return "Unknown Student"
    match = re.search(r'Name\s*[\:\.]*\s*([A-Za-z\s\.]+?)(?=Registration|Regn|Roll|\d{3}\-|\n|$)', text, re.IGNORECASE)
    if match:
        raw_name = re.sub(r'[^A-Za-z\s\.]', '', match.group(1)).strip()
        if len(raw_name) > 2: return raw_name
    return "Unknown Student"

def extract_course(text: str):
    if not text: return "B.A. (Honours) Examination"
    match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', text, re.IGNORECASE)
    if match:
        raw_course = match.group(0).strip()
        cleaned_course = re.sub(r'Semester\s*[\-\–\_]?\s*(VI|V|IV|III|II|I|\d+)', '', raw_course, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', cleaned_course).strip()
    return "B.A. (Honours) Examination"

def parse_summary_table_local(table_text: str):
    sems = []
    cgpa = "N.A."
    grade = "Fail / Semester Not Cleared"
    if not table_text: return sems, cgpa, grade

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', table_text)
    is_not_cleared = bool(re.search(r'(?:Semester\s*not\s*cleared|not\s*cleared)', text_clean, re.IGNORECASE))

    lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
    sem_keys = ['I', 'II', 'III', 'IV', 'V', 'VI']

    for line in lines:
        for sem in sem_keys:
            if re.search(rf'\b{sem}\b', line):
                years = re.findall(r'\b(20\d\d)\b', line)
                integers = [i for i in re.findall(r'\b(\d{2,3})\b', line) if i not in years]
                floats = re.findall(r'\b([1-9]\.\d{2,3})\b', line)

                if years and integers and floats:
                    s_yr = years[0]
                    s_fm = integers[0] if len(integers) >= 1 else "400"
                    s_marks = integers[1] if len(integers) >= 2 else (integers[0] if integers else "-")
                    s_cred = integers[2] if len(integers) >= 3 else "20"
                    s_sgpa = floats[0]

                    if not any(item["semester"] == sem for item in sems):
                        sems.append({
                            "semester": sem,
                            "year": s_yr,
                            "full_marks": s_fm,
                            "marks": s_marks,
                            "credit": s_cred,
                            "sgpa": s_sgpa
                        })

    sem_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
    sems = sorted(sems, key=lambda x: sem_order.get(x["semester"], 99))

    if not is_not_cleared:
        all_floats = re.findall(r'\b([1-9]\.\d{2,3})\b', text_clean)
        valid_gpas = [f for f in all_floats if 1.0 <= float(f) <= 10.0]
        if valid_gpas: cgpa = valid_gpas[-1]

        grade_m = re.search(r'\b(A\+|B\+|C\+|A|B|C|D|O)(?!\w)', text_clean)
        if grade_m: grade = grade_m.group(1)

    return sems, cgpa, grade

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Shyampur Siddheswari Mahavidyalaya Marksheet Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- SAFE STARTUP DB CREATION & MIGRATION ---
@app.on_event("startup")
def startup_db_setup():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.begin() as conn:
            if "postgresql" in DATABASE_URL:
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS course VARCHAR DEFAULT 'Unknown Course';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_status VARCHAR DEFAULT 'Unemployed';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_details VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS proof_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS full_marks VARCHAR DEFAULT '400';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS credit VARCHAR DEFAULT '20';"))
            elif "sqlite" in DATABASE_URL:
                columns = [row[1] for row in conn.execute(text("PRAGMA table_info(students);")).fetchall()]
                if columns:
                    if "course" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN course VARCHAR DEFAULT 'Unknown Course';"))
                    if "marksheet_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_received BOOLEAN DEFAULT 0;"))
                    if "certificate_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_received BOOLEAN DEFAULT 0;"))
                    if "post_grad_status" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_status VARCHAR DEFAULT 'Unemployed';"))
                    if "post_grad_details" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_details VARCHAR;"))
                    if "proof_document_path" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN proof_document_path VARCHAR;"))
                
                sem_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(semester_records);")).fetchall()]
                if sem_columns:
                    if "full_marks" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN full_marks VARCHAR DEFAULT '400';"))
                    if "credit" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN credit VARCHAR DEFAULT '20';"))
    except Exception as e:
        print(f"Startup Migration Note: {e}")

# --- FRONTEND ROUTE ---
@app.get("/")
def serve_frontend(username: str = Depends(authenticate_admin)):
    return FileResponse("index.html")

# --- API ENDPOINTS ---

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(
    file: UploadFile = File(...), 
    selected_course: str = Form("AUTO"),
    db: Session = Depends(get_db), 
    user: str = Depends(authenticate_admin)
):
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_count = 0
    debug_logs = []
    
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        ai_quota_exceeded = False
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # --- STRATEGY 1: OPENAI VISION WITH AUTO-RETRY RATE-LIMITER ---
                reg_no = None
                normalized_semesters = []
                
                if ai_client and not ai_quota_exceeded:
                    try:
                        # 0.4s pause between pages prevents hitting OpenAI RPM limits on 100+ page PDFs
                        if page_num > 0:
                            time.sleep(0.4)

                        data = parse_marksheet_with_openai_vision(page)

                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            debug_logs.append(f"Page {page_num+1}: Registration Number not found.")
                            continue

                        roll_no = data.get("roll_no", "Unknown")
                        name = data.get("name", "Unknown Student")
                        course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "B.A. (Honours) Examination (Under CBCS)")
                        overall_cgpa = data.get("overall_cgpa", "N.A.")
                        overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")
                        
                        raw_semesters = data.get("semesters", [])
                        for sem in raw_semesters:
                            if not isinstance(sem, dict): continue
                            raw_s = str(sem.get("semester") or sem.get("sem") or "").strip().upper()
                            raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                            if not raw_s: continue

                            normalized_semesters.append({
                                "semester": raw_s,
                                "year": str(sem.get("year") or sem.get("exam_year") or "").strip(),
                                "full_marks": str(sem.get("full_marks") or sem.get("total_marks") or "400").strip(),
                                "marks": str(sem.get("marks") or sem.get("marks_obtained") or sem.get("obtained") or "-").strip(),
                                "credit": str(sem.get("credit") or sem.get("semester_credit") or "20").strip(),
                                "sgpa": str(sem.get("sgpa") or sem.get("gpa") or "N.A.").strip()
                            })
                    except Exception as ai_err:
                        debug_logs.append(f"Page {page_num+1} AI Exception: {ai_err}")

                # --- STRATEGY 2: LOCAL PYMUPDF + TESSERACT OCR FALLBACK ---
                if not ai_client or ai_quota_exceeded or not reg_no:
                    full_text = page.get_text("text") or ""
                    reg_no, match_method = extract_reg_no_bulletproof(full_text)

                    # If native text is empty (scanned page), run Tesseract OCR if available
                    if not reg_no and pytesseract:
                        try:
                            pix = page.get_pixmap(dpi=110)
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            gray = img.convert('L')
                            ocr_text = pytesseract.image_to_string(gray, lang="eng")
                            if ocr_text:
                                full_text = ocr_text
                                reg_no, match_method = extract_reg_no_bulletproof(full_text)
                        except Exception as ocr_err:
                            debug_logs.append(f"Page {page_num+1} OCR Exception: {ocr_err}")

                    if not reg_no:
                        debug_logs.append(f"Page {page_num+1}: Local fallback could not detect Registration No.")
                        continue

                    roll_no = extract_roll_no_bulletproof(full_text)
                    name = extract_name_bulletproof(full_text)
                    course = selected_course if (selected_course and selected_course != "AUTO") else extract_course(full_text)

                    rect = page.rect
                    table_rect = fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)
                    table_text = extract_text_rows_from_rect(page, table_rect)

                    normalized_semesters, overall_cgpa, overall_grade = parse_summary_table_local(table_text)

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    
                    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, 
                            semester=sem["semester"], 
                            year=sem["year"], 
                            full_marks=sem["full_marks"],
                            marks_obtained=sem["marks"], 
                            credit=sem["credit"],
                            sgpa=sem["sgpa"]
                        ))
                else:
                    admission_year = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    student = Student(
                        registration_no=reg_no, 
                        roll_no=roll_no, 
                        name=name, 
                        admission_year=admission_year, 
                        course=course, 
                        overall_cgpa=overall_cgpa, 
                        overall_grade=overall_grade
                    )
                    db.add(student)
                    for sem in normalized_semesters:
                        db.add(SemesterRecord(
                            registration_no=reg_no, 
                            semester=sem["semester"], 
                            year=sem["year"], 
                            full_marks=sem["full_marks"],
                            marks_obtained=sem["marks"], 
                            credit=sem["credit"],
                            sgpa=sem["sgpa"]
                        ))
                        
                extracted_count += 1
                method_label = "OpenAI AI Vision" if (ai_client and not ai_quota_exceeded and reg_no) else "Local Fallback"
                debug_logs.append(f"Page {page_num+1} Success ({method_label}): Reg={reg_no}, Name={name}, Roll={roll_no}, CGPA={overall_cgpa}, Grade={overall_grade}, Sems={len(normalized_semesters)}")
            except Exception as page_err:
                db.rollback()
                debug_logs.append(f"Page {page_num+1} Exception: {page_err}")
                continue

        if doc: doc.close()
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_pdf_path): 
            os.remove(temp_pdf_path)

    return {
        "message": f"Successfully extracted {extracted_count} student record(s)!",
        "debug_logs": debug_logs
    }

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student record not found.")
    
    sems = db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).all()
    
    passout_year = "Unknown"
    for s in sems:
        if s.semester in ["VI", "6", "VI."] and s.year:
            passout_year = s.year

    return {
        "student": {
            "name": student.name, 
            "reg_no": student.registration_no, 
            "roll_no": student.roll_no,
            "admission_year": student.admission_year, 
            "passout_year": passout_year, 
            "course": student.course,
            "cgpa": student.overall_cgpa, 
            "grade": student.overall_grade,
            "marksheet_received": student.marksheet_received, 
            "certificate_received": student.certificate_received,
            "status": student.post_grad_status, 
            "details": student.post_grad_details, 
            "proof": student.proof_document_path
        },
        "semesters": [
            {
                "semester": s.semester, 
                "year": s.year, 
                "full_marks": s.full_marks or "400",
                "marks": s.marks_obtained, 
                "credit": s.credit or "20",
                "sgpa": s.sgpa
            } 
            for s in sems
        ]
    }

@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    db.delete(student)
    db.commit()
    return {"message": f"Student {reg_no} deleted successfully"}

@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    db.query(SemesterRecord).delete()
    db.query(Student).delete()
    db.commit()
    return {"message": "All student records cleared successfully"}

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(
    reg_no: str,
    course: str = Form(...),
    marksheet_received: bool = Form(...),
    certificate_received: bool = Form(...),
    status: str = Form(...),
    details: str = Form(""),
    proof_file: UploadFile = File(None),
    db: Session = Depends(get_db),
    user: str = Depends(authenticate_admin)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student record not found.")

    student.course = course
    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.post_grad_status = status
    student.post_grad_details = details

    if proof_file:
        safe_filename = f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}"
        file_path = os.path.join("uploads", safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path

    db.commit()
    return {"message": "Record updated successfully!"}

@app.get("/api/admin/grade-stats")
def get_grade_stats(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    stats = db.query(Student.course, Student.overall_grade, func.count(Student.registration_no)) \
              .group_by(Student.course, Student.overall_grade).all()
    
    result = {}
    for course, grade, count in stats:
        c = course if course else "Unknown Course"
        g = grade if grade else "Fail/NA"
        if c not in result:
            result[c] = {}
        result[c][g] = count
        
    return result

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    return db.query(Student).all()

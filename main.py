import os
import re
import shutil
import secrets
import json
import base64
import time
import gc
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

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
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
    remarks = Column(String, default="Qualified")
    
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    post_grad_status = Column(String, default="Unknown") 
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

# --- STRICT OPENAI GPT-4o-MINI VISION PARSER ---

def parse_marksheet_with_openai_vision(page):
    rect = page.rect

    # 1. Full Page Image for Header Details
    pix_full = page.get_pixmap(dpi=120)
    img_bytes_full = pix_full.tobytes("jpeg")
    base64_full = base64.b64encode(img_bytes_full).decode('utf-8')

    prompt_header = """
    Extract header information from this Calcutta University Marksheet image and return ONLY a JSON object:
    {
      "registration_no": "424-1215-0072-20",
      "roll_no": "202424-11-0427",
      "name": "TANISHA PARVIN",
      "course": "B.A. (Honours) Examination (Under CBCS)"
    }
    STRICT RULES:
    1. Registration No format: 3-4-4-2 (e.g. 424-1215-0072-20).
    2. Roll No format: 6-2-4 (e.g. 202424-11-0427). Double-check digits with 100% precision.
    3. Omit 'Semester - VI' or 'Semester - I' from course title.
    """

    # 2. Cropped Summary Table Image (Y: 46% to 88% height)
    table_rect = fitz.Rect(0, rect.height * 0.46, rect.width, rect.height * 0.88)
    pix_table = page.get_pixmap(dpi=150, clip=table_rect)
    img_bytes_table = pix_table.tobytes("jpeg")
    base64_table = base64.b64encode(img_bytes_table).decode('utf-8')

    prompt_table = """
    Analyze ONLY this cropped Semester Summary Table from the bottom of a Calcutta University Grade Sheet and return ONLY a JSON object:

    {
      "overall_cgpa": "N.A.",
      "overall_grade": "Fail / Semester Not Cleared",
      "remarks": "Semester not cleared",
      "semesters": [
        {"semester": "I", "year": "2020", "full_marks": "400", "marks": "336", "credit": "20", "sgpa": "8.068"},
        {"semester": "II", "year": "2021", "full_marks": "400", "marks": "335", "credit": "20", "sgpa": "8.314"},
        {"semester": "III", "year": "2021", "full_marks": "500", "marks": "379", "credit": "26", "sgpa": "7.540"}
      ]
    }

    STRICT CRITICAL RULES FOR LETTER GRADE AND CGPA:
    1. LETTER GRADE COLUMN RULE: Look ONLY at the 'Letter Grade' column in the summary table (last column).
       - If the 'Letter Grade' cell is BLANK / EMPTY, or if 'Semester not cleared' is in Remarks:
         - overall_grade MUST BE "Fail / Semester Not Cleared"
         - overall_cgpa MUST BE "N.A."
       - ONLY if the 'Letter Grade' cell contains an explicit letter grade (e.g. B+, A+, A, B, C+, C, D, O):
         - overall_grade = that exact letter grade (e.g. "B+", "A", "A+")
         - overall_cgpa = the exact number in the 'CGPA' column on row VI (e.g. "6.819", "7.311", "7.378").
    2. NEVER invent a Letter Grade or CGPA if the cell in the table is blank/empty!
    3. ONLY PRESENT SEMESTERS: Extract ONLY rows that contain marks and SGPA in the table. Do NOT include empty/blank semester rows (e.g., if Sem IV, V, VI are blank, return ONLY Sem I, II, III).
    4. REMARKS: Read the exact Remarks line right below the table (e.g. 'Qualified with Honours' or 'Semester not cleared').
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_table = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_table},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_table}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            table_data = json.loads(response_table.choices[0].message.content)

            response_header = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_header},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_full}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ]
            )
            header_data = json.loads(response_header.choices[0].message.content)

            return {
                "registration_no": header_data.get("registration_no"),
                "roll_no": header_data.get("roll_no"),
                "name": header_data.get("name"),
                "course": header_data.get("course"),
                "overall_cgpa": table_data.get("overall_cgpa", "N.A."),
                "overall_grade": table_data.get("overall_grade", "Fail / Semester Not Cleared"),
                "remarks": table_data.get("remarks", "Qualified with Honours"),
                "semesters": table_data.get("semesters", [])
            }
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
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
    remarks = "Qualified"
    if not table_text: return sems, cgpa, grade, remarks

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', table_text)
    is_not_cleared = bool(re.search(r'(?:Semester\s*not\s*cleared|not\s*cleared)', text_clean, re.IGNORECASE))

    rem_match = re.search(r'Remarks\s*[\:\.]*\s*([^\n\r]+)', text_clean, re.IGNORECASE)
    if rem_match:
        remarks = rem_match.group(1).strip()

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

    return sems, cgpa, grade, remarks

# --- BACKGROUND WORKER FOR LARGE PDFs ---

def process_large_pdf_in_background(temp_pdf_path: str, selected_course: str):
    db = SessionLocal()
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        ai_quota_exceeded = False
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                reg_no = None
                normalized_semesters = []
                remarks = "Qualified"
                
                # --- STRATEGY 1: OPENAI VISION ---
                if ai_client and not ai_quota_exceeded:
                    try:
                        if page_num > 0:
                            time.sleep(0.3)

                        data = parse_marksheet_with_openai_vision(page)

                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            continue

                        roll_no = data.get("roll_no", "Unknown")
                        name = data.get("name", "Unknown Student")
                        course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "B.A. (Honours) Examination (Under CBCS)")
                        
                        remarks = data.get("remarks", "Qualified with Honours")
                        overall_cgpa = data.get("overall_cgpa", "N.A.")
                        overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")

                        # Python-level enforcement of the Letter Grade rule
                        if "not cleared" in str(remarks).lower() or overall_grade.lower() in ["none", "null", "n.a.", ""]:
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

                            # STRICT FILTER: Skip empty/blank semester rows that have no year and no SGPA
                            if (not yr_val or yr_val in ["-", "N.A.", "None", ""]) and (not sgpa_val or sgpa_val in ["-", "N.A.", "None", ""]):
                                continue

                            normalized_semesters.append({
                                "semester": raw_s,
                                "year": yr_val if yr_val not in ["-", "N.A."] else "-",
                                "full_marks": str(sem.get("full_marks") or sem.get("total_marks") or "400").strip(),
                                "marks": marks_val if marks_val not in ["-", "N.A."] else "-",
                                "credit": str(sem.get("credit") or sem.get("semester_credit") or "20").strip(),
                                "sgpa": sgpa_val if sgpa_val not in ["-", "N.A."] else "N.A."
                            })
                    except Exception as ai_err:
                        if "429" in str(ai_err) or "rate" in str(ai_err) or "quota" in str(ai_err):
                            ai_quota_exceeded = True

                # --- STRATEGY 2: LOCAL FALLBACK ---
                if not ai_client or ai_quota_exceeded or not reg_no:
                    full_text = page.get_text("text") or ""
                    reg_no, match_method = extract_reg_no_bulletproof(full_text)
                    if not reg_no:
                        continue

                    roll_no = extract_roll_no_bulletproof(full_text)
                    name = extract_name_bulletproof(full_text)
                    course = selected_course if (selected_course and selected_course != "AUTO") else extract_course(full_text)

                    rect = page.rect
                    table_rect = fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)
                    table_text = extract_text_rows_from_rect(page, table_rect)

                    normalized_semesters, overall_cgpa, overall_grade, remarks = parse_summary_table_local(table_text)

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    existing_student.remarks = remarks
                    
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
                        overall_grade=overall_grade,
                        remarks=remarks
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

                if page_num % 5 == 0:
                    db.commit()
                    gc.collect()

            except Exception as page_err:
                db.rollback()
                continue

        if doc: doc.close()
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Background PDF error: {e}")
    finally:
        db.close()
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

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
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS remarks VARCHAR DEFAULT 'Qualified';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_status VARCHAR DEFAULT 'Unknown';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_details VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS proof_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS full_marks VARCHAR DEFAULT '400';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS credit VARCHAR DEFAULT '20';"))
                conn.execute(text("UPDATE students SET post_grad_status = 'Unknown' WHERE post_grad_status = 'Unemployed';"))
            elif "sqlite" in DATABASE_URL:
                columns = [row[1] for row in conn.execute(text("PRAGMA table_info(students);")).fetchall()]
                if columns:
                    if "course" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN course VARCHAR DEFAULT 'Unknown Course';"))
                    if "remarks" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN remarks VARCHAR DEFAULT 'Qualified';"))
                    if "marksheet_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_received BOOLEAN DEFAULT 0;"))
                    if "certificate_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_received BOOLEAN DEFAULT 0;"))
                    if "post_grad_status" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_status VARCHAR DEFAULT 'Unknown';"))
                    if "post_grad_details" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN post_grad_details VARCHAR;"))
                    if "proof_document_path" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN proof_document_path VARCHAR;"))
                
                sem_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(semester_records);")).fetchall()]
                if sem_columns:
                    if "full_marks" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN full_marks VARCHAR DEFAULT '400';"))
                    if "credit" not in sem_columns: conn.execute(text("ALTER TABLE semester_records ADD COLUMN credit VARCHAR DEFAULT '20';"))
                conn.execute(text("UPDATE students SET post_grad_status = 'Unknown' WHERE post_grad_status = 'Unemployed';"))
    except Exception as e:
        print(f"Startup Migration Note: {e}")

# --- FRONTEND ROUTE ---
@app.get("/")
def serve_frontend(username: str = Depends(authenticate_admin)):
    return FileResponse("index.html")

# --- API ENDPOINTS ---

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    selected_course: str = Form("AUTO"),
    user: str = Depends(authenticate_admin)
):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()

    background_tasks.add_task(process_large_pdf_in_background, temp_pdf_path, selected_course)

    return {
        "message": f"🚀 Successfully started background processing for {total_pages} page(s)! You can open Tab 6 (Student Directory) to see extracted records in real-time."
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
            "remarks": student.remarks or "Qualified",
            "marksheet_received": student.marksheet_received, 
            "certificate_received": student.certificate_received,
            "status": student.post_grad_status or "Unknown", 
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

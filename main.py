import os
import re
import shutil
import secrets
import gc
from PIL import Image
import numpy as np
import cv2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

import pytesseract

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

# --- DATABASE SETUP (AUTO-RECONNECT SSL PING) ---
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

# --- OPENCV OTSU ADAPTIVE BINARIZATION ---

def opencv_erase_watermark_otsu(pil_img):
    """
    Applies Otsu's Adaptive Binarization to calculate optimal thresholding
    for yellowed or warm-lit scanned marksheet papers automatically.
    """
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

# --- HEADER PARSERS ---

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

    pattern_match = re.search(r'\b(\d{3})[\-\s\.\/]*(\d{4})[\-\s\.\/]*(\d{4})[\-\s\.\/]*(\d{2})\b', t_fixed)
    if pattern_match:
        g = pattern_match.groups()
        return f"{g[0]}-{g[1]}-{g[2]}-{g[3]}", "3-4-4-2 Pattern"

    all_digits_clusters = re.findall(r'\b\d{13}\b', re.sub(r'\D', ' ', t_fixed))
    if all_digits_clusters:
        d = all_digits_clusters[0]
        return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}", "13-Digit Cluster"

    return None, f"No 13-digit pattern found (Text Len={len(text)})"

def extract_roll_no_bulletproof(text: str):
    if not text: return "Unknown"
    t = re.sub(r'[—–_~]', '-', text)
    t_fixed = t.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1').replace('S', '5')

    match = re.search(r'Roll[^\d]{0,15}([0-9\-\s\.\/]{12,20})', t_fixed, re.IGNORECASE)
    if match:
        digits = re.sub(r'\D', '', match.group(1))
        if len(digits) >= 12:
            d = digits[:12]
            return f"{d[:6]}-{d[6:8]}-{d[8:]}"

    pattern_match = re.search(r'\b(\d{6})[\-\s\.\/]*(\d{2})[\-\s\.\/]*(\d{4})\b', t_fixed)
    if pattern_match:
        g = pattern_match.groups()
        return f"{g[0]}-{g[1]}-{g[2]}"

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
        if len(raw_name) > 2:
            return raw_name
    return "Unknown Student"

def extract_course(text: str):
    if not text: return "B.A. (Honours) Examination"
    match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', text, re.IGNORECASE)
    if match:
        raw_course = match.group(0).strip()
        # Clean out "Semester - VI" or "Semester-VI" or "Semester 6"
        cleaned_course = re.sub(r'Semester\s*[\-\–\_]?\s*(VI|V|IV|III|II|I|\d+)', '', raw_course, flags=re.IGNORECASE)
        cleaned_course = re.sub(r'\s+', ' ', cleaned_course).strip()
        return cleaned_course
    return "B.A. (Honours) Examination"

# --- UNIVERSAL SEMESTER TABLE PARSER ---

def extract_all_semesters(text: str):
    sems_data = []
    if not text: return sems_data

    clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', text)
    clean = clean.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')

    # Matches Roman Numeral + Year + Full Marks + Marks Obtained + Credit + SGPA
    sem_pattern = re.compile(
        r'\b(I|II|III|IV|V|VI)\b[\s\n\r]+(\d{4})[\s\n\r]+(\d{3})[\s\n\r]+(\d{2,3})[\s\n\r]+(\d{2})[\s\n\r]+([0-9]\.\d{2,3})',
        re.IGNORECASE
    )

    for m in sem_pattern.finditer(clean):
        s_num, s_yr, s_fm, s_marks, s_cred, s_sgpa = m.groups()
        s_num_upper = s_num.upper()
        if not any(x["semester"] == s_num_upper for x in sems_data):
            sems_data.append({
                "semester": s_num_upper,
                "year": s_yr,
                "full_marks": s_fm,
                "marks": s_marks,
                "credit": s_cred,
                "sgpa": s_sgpa
            })

    # Line-by-line fallback if multiline pattern missed rows
    if len(sems_data) < 5:
        sem_keys = ['I', 'II', 'III', 'IV', 'V', 'VI']
        for line in clean.splitlines():
            line_str = line.strip()
            for sem in sem_keys:
                if re.search(rf'\b{sem}\b', line_str):
                    years = re.findall(r'\b(20\d\d)\b', line_str)
                    floats = re.findall(r'\b([1-9]\.\d{2,3})\b', line_str)
                    integers = [i for i in re.findall(r'\b(\d{2,3})\b', line_str) if i not in years]

                    if years and floats:
                        s_yr = years[0]
                        s_sgpa = floats[0]
                        s_fm = integers[0] if len(integers) >= 1 else "400"
                        s_marks = integers[1] if len(integers) >= 2 else (integers[0] if integers else "-")
                        s_cred = integers[2] if len(integers) >= 3 else "20"

                        if not any(x["semester"] == sem for x in sems_data):
                            sems_data.append({
                                "semester": sem,
                                "year": s_yr,
                                "full_marks": s_fm,
                                "marks": s_marks,
                                "credit": s_cred,
                                "sgpa": s_sgpa
                            })

    # Sort semesters in order: I, II, III, IV, V, VI
    sem_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
    sems_data = sorted(sems_data, key=lambda x: sem_order.get(x["semester"], 99))

    return sems_data

def parse_ocr_summary_table(ocr_text: str):
    sems_data = extract_all_semesters(ocr_text)
    overall_cgpa = "N.A."
    overall_grade = "Fail / Semester Not Cleared"

    if not ocr_text:
        return sems_data, overall_cgpa, overall_grade

    clean_ocr = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', ocr_text)

    # 1. Check Remarks
    is_failed = bool(re.search(r'Semester\s*not\s*cleared|not\s*cleared', clean_ocr, re.IGNORECASE))
    if is_failed:
        overall_cgpa = "N.A."
        overall_grade = "Fail / Semester Not Cleared"
    else:
        # On row VI: extract CGPA and Grade
        vi_match = re.search(r'\bVI\b\s+(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+(\d+)\s+([\d\.]+)\s+([A-Z\+\-]+)', clean_ocr)
        if vi_match:
            overall_cgpa = vi_match.group(7) # 6.819
            grade_val = vi_match.group(8) # B+
            if grade_val.upper() in ["A+", "A", "B+", "B", "C+", "C", "D", "O"]:
                overall_grade = grade_val
        else:
            all_gpas = re.findall(r'\b([1-9]\.\d{2,3})\b', clean_ocr)
            if len(all_gpas) >= 7:
                overall_cgpa = all_gpas[-1]  # Last float is CGPA (6.819)
            elif len(all_gpas) >= 1:
                cg_m = re.search(r'140\s+([\d\.]+)|CGPA[^\d]{0,15}([\d\.]+)', clean_ocr)
                if cg_m:
                    overall_cgpa = cg_m.group(1) or cg_m.group(2)

            grade_m = re.search(r'\b(A\+|B\+|C\+|A|B|C|D|O)(?!\w)', clean_ocr)
            if grade_m:
                overall_grade = grade_m.group(1)

    return sems_data, overall_cgpa, overall_grade

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
    try:
        yield db
    finally:
        db.close()

# --- SAFE STARTUP DB CREATION & MIGRATION ---
@app.on_event("startup")
def startup_db_setup():
    try:
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            if "postgresql" in DATABASE_URL:
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS course VARCHAR DEFAULT 'Unknown Course';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_status VARCHAR DEFAULT 'Unemployed';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS post_grad_details VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS proof_document_path VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS full_marks VARCHAR DEFAULT '400';"))
                conn.execute(text("ALTER TABLE semester_records ADD COLUMN IF NOT EXISTS credit VARCHAR DEFAULT '20';"))
                conn.commit()
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
                conn.commit()
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
    db: Session = Depends(get_db), 
    user: str = Depends(authenticate_admin)
):
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_count = 0
    debug_logs = []
    
    try:
        if not fitz:
            raise HTTPException(status_code=500, detail="PyMuPDF library is not installed.")

        doc = fitz.open(temp_pdf_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                rect = page.rect
                
                # 1. Native Header Text
                full_text = page.get_text("text") or ""
                reg_no, match_method = extract_reg_no_bulletproof(full_text)

                if not reg_no:
                    try:
                        pix_full = page.get_pixmap(dpi=110)
                        img_full = Image.frombytes("RGB", [pix_full.width, pix_full.height], pix_full.samples)
                        gray = img_full.convert('L')
                        ocr_text = pytesseract.image_to_string(gray, lang="eng")
                        if ocr_text:
                            full_text = ocr_text
                            reg_no, match_method = extract_reg_no_bulletproof(full_text)
                    except Exception as ocr_err:
                        debug_logs.append(f"Page {page_num+1} OCR Exception: {ocr_err}")

                if not reg_no:
                    debug_logs.append(f"Page {page_num+1} Failed: {match_method}")
                    continue

                roll_no = extract_roll_no_bulletproof(full_text)
                name = extract_name_bulletproof(full_text)
                course = extract_course(full_text)

                # 2. CROP SUMMARY TABLE REGION & Otsu Binarize
                table_rect = fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)
                table_text = page.get_text("text", clip=table_rect) or ""

                # Parse native text first
                sems_data, overall_cgpa, overall_grade = parse_ocr_summary_table(table_text)

                # If native text missed semester rows, apply OpenCV Otsu OCR
                if len(sems_data) < 5:
                    try:
                        pix_table = page.get_pixmap(dpi=140, clip=table_rect)
                        img_table = Image.frombytes("RGB", [pix_table.width, pix_table.height], pix_table.samples)
                        
                        clean_table_img = opencv_erase_watermark_otsu(img_table)
                        table_ocr_text = pytesseract.image_to_string(clean_table_img, lang="eng", config="--psm 6")
                        
                        sems_ocr, cgpa_ocr, grade_ocr = parse_ocr_summary_table(table_ocr_text)
                        if len(sems_ocr) > len(sems_data):
                            sems_data = sems_ocr
                        if cgpa_ocr != "N.A.":
                            overall_cgpa = cgpa_ocr
                            overall_grade = grade_ocr
                    except Exception as table_ocr_err:
                        debug_logs.append(f"Page {page_num+1} Table OCR Exception: {table_ocr_err}")

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name if name != "Unknown Student" else existing_student.name
                    existing_student.roll_no = roll_no if roll_no != "Unknown" else existing_student.roll_no
                    existing_student.course = course
                    existing_student.overall_cgpa = overall_cgpa
                    existing_student.overall_grade = overall_grade
                    
                    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                    for sem in sems_data:
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
                    for sem in sems_data:
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
                debug_logs.append(f"Page {page_num+1} Success ({match_method}): Reg={reg_no}, Name={name}, Roll={roll_no}, CGPA={overall_cgpa}, Grade={overall_grade}, Sems={len(sems_data)}")
            except Exception as page_err:
                db.rollback()
                debug_logs.append(f"Page {page_num+1} DB Exception: {page_err}")
                continue

        doc.close()
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
    
    # Calculate passout year: Exam year of Semester VI
    passout_year = "Unknown"
    for s in sems:
        if s.semester == "VI" and s.year:
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

import os
import re
import shutil
import secrets
import gc
from PIL import Image

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
    if not text: return "B.A. (Honours)"
    match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    return "B.A. (Honours)"

# --- TARGETED SUMMARY TABLE REGION PARSER ---

def parse_summary_table(table_text: str):
    sems = []
    cgpa = "N.A."
    grade = "Fail / Semester Not Cleared"
    if not table_text:
        return sems, cgpa, grade

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', table_text)

    # 1. Check if Semester Not Cleared
    is_not_cleared = bool(re.search(r'(?:Semester\s*not\s*cleared|not\s*cleared)', text_clean, re.IGNORECASE))

    # 2. Extract All 6 Semester Rows
    # Format: Roman Numeral + Year (4 digits) + Full Marks (3 digits) + Marks Obtained (2-3 digits) + Semester Credit (2 digits) + SGPA (float)
    row_pattern = re.compile(
        r'\b(I|II|III|IV|V|VI)\b\s+(\d{4})\s+(\d{3})\s+(\d{2,3})\s+(\d{2})\s+([\d\.]+)',
        re.IGNORECASE
    )

    for m in row_pattern.finditer(text_clean):
        s_num, s_yr, s_fm, s_marks, s_cred, s_sgpa = m.groups()
        sems.append({
            "semester": s_num.upper(),
            "year": s_yr,
            "full_marks": s_fm,
            "marks": s_marks,
            "credit": s_cred,
            "sgpa": s_sgpa
        })

    # 3. Extract CGPA and Grade specifically from Semester VI row or table
    if not is_not_cleared:
        # Semester VI Row with CGPA and Grade: VI 2024 400 270 24 6.686 140 6.819 B+
        vi_match = re.search(
            r'\bVI\b\s+(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+(\d+)\s+([\d\.]+)\s+([A-Z\+\-]+)',
            text_clean
        )
        if vi_match:
            cgpa = vi_match.group(7) # 6.819
            grade_val = vi_match.group(8) # B+
            if grade_val.upper() in ["A+", "A", "B+", "B", "C+", "C", "D", "O"]:
                grade = grade_val
        else:
            # Fallback CGPA search (Last float in table)
            gpas = re.findall(r'\b([1-9]\.\d{2,3})\b', text_clean)
            valid_gpas = [g for g in gpas if 1.0 <= float(g) <= 10.0]
            if valid_gpas:
                cgpa = valid_gpas[-1]
            
            # Fallback Grade search (Excludes status code P)
            grade_m = re.search(r'\b(A\+|A|B\+|B|C\+|C|D|O)\b', text_clean)
            if grade_m:
                grade = grade_m.group(1)

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
                
                # 1. Full Page Text for Header Info
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

                # 2. CROP SUMMARY TABLE REGION (48% to 88% page height)
                table_rect = fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)
                table_text = page.get_text("text", clip=table_rect) or ""

                if len(table_text.strip()) < 20:
                    try:
                        pix_table = page.get_pixmap(dpi=150, clip=table_rect)
                        img_table = Image.frombytes("RGB", [pix_table.width, pix_table.height], pix_table.samples)
                        gray_table = img_table.convert('L')
                        table_text = pytesseract.image_to_string(gray_table, lang="eng", config="--psm 6")
                    except Exception as table_ocr_err:
                        debug_logs.append(f"Page {page_num+1} Table OCR Exception: {table_ocr_err}")

                # 3. Parse Semesters, CGPA, and Grade strictly from Table Region
                sems_data, overall_cgpa, overall_grade = parse_summary_table(table_text)

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

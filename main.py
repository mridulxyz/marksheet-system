import os
import re
import shutil
import secrets
import gc
from PIL import Image

try:
    import pypdf
except ImportError:
    pypdf = None

import pdfplumber
import pytesseract

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func
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

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
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
    marks_obtained = Column(String)
    sgpa = Column(String)
    
    student = relationship("Student", back_populates="semesters")

Base.metadata.create_all(bind=engine)
os.makedirs("uploads", exist_ok=True)

# --- FASTAPI APP SETUP ---
app = FastAPI(title="University Admin Portal")

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

# --- HELPER: Binarize image to erase background watermark ---
def clean_image_for_ocr(pil_img):
    gray = pil_img.convert('L')
    bw = gray.point(lambda p: 255 if p > 160 else 0) # Erase gray watermark background
    return bw

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
    try:
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    # 1. Native Digital Text Extraction
                    text = page.extract_text() or ""
                    
                    # 2. Scanned PDF Fallback: Fast binarized OCR if text is less than 30 chars
                    if len(text.strip()) < 30:
                        print(f"Page {page_num+1}: Native text empty. Triggering fast OCR...")
                        try:
                            raw_img = page.to_image(resolution=100).original
                            cleaned_img = clean_image_for_ocr(raw_img)
                            text = pytesseract.image_to_string(cleaned_img, lang="eng", config="--psm 6")
                            del raw_img, cleaned_img
                            gc.collect()
                        except Exception as ocr_err:
                            print(f"OCR Error on page {page_num+1}: {ocr_err}")
                            continue

                    text_clean = re.sub(r'[—–_~]', '-', text)

                    # 3. Registration Number Extraction (With digit sanitization)
                    reg_no = None
                    reg_match = re.search(r'(?:Registration|Regn|Reg)[\s\.\:\-]*No[\.\:\s]*([0-9OoQIl\-\s\.\/]+)', text_clean, re.IGNORECASE)
                    if reg_match:
                        raw_val = reg_match.group(1).replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1')
                        digits = re.sub(r'\D', '', raw_val)
                        if len(digits) >= 13:
                            digits = digits[:13]
                            reg_no = f"{digits[:3]}-{digits[3:7]}-{digits[7:11]}-{digits[11:]}"

                    # Fallback pattern search for CU Registration: XXX-XXXX-XXXX-XX
                    if not reg_no:
                        text_digit_fixed = text_clean.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1')
                        cu_pattern = re.search(r'\b(\d{3})[\-\s\.\/]*(\d{4})[\-\s\.\/]*(\d{4})[\-\s\.\/]*(\d{2})\b', text_digit_fixed)
                        if cu_pattern:
                            g = cu_pattern.groups()
                            reg_no = f"{g[0]}-{g[1]}-{g[2]}-{g[3]}"

                    if not reg_no:
                        print(f"Page {page_num+1}: Registration number not found.")
                        continue 

                    # 4. Roll Number Extraction
                    roll_no = "Unknown"
                    roll_match = re.search(r'Roll[\s\&\.]*No[\.\:\s]*([0-9OoQIl\-\s\.\/]+)', text_clean, re.IGNORECASE)
                    if roll_match:
                        raw_val = roll_match.group(1).replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1')
                        digits = re.sub(r'\D', '', raw_val)
                        if len(digits) >= 12:
                            digits = digits[:12]
                            roll_no = f"{digits[:6]}-{digits[6:8]}-{digits[8:]}"

                    if roll_no == "Unknown":
                        text_digit_fixed = text_clean.replace('O', '0').replace('o', '0').replace('Q', '0').replace('I', '1').replace('l', '1')
                        cu_roll_p = re.search(r'\b(\d{6})[\-\s\.\/]*(\d{2})[\-\s\.\/]*(\d{4})\b', text_digit_fixed)
                        if cu_roll_p:
                            g = cu_roll_p.groups()
                            roll_no = f"{g[0]}-{g[1]}-{g[2]}"

                    # 5. Student Name Extraction
                    name = "Unknown Student"
                    name_match = re.search(r'Name\s*[\:\.]*\s*([A-Za-z\s\.]+?)(?=Registration|Regn|Roll|\d{3}\-|\n|$)', text_clean, re.IGNORECASE)
                    if name_match:
                        raw_name = re.sub(r'[^A-Za-z\s\.]', '', name_match.group(1)).strip()
                        if len(raw_name) > 2:
                            name = raw_name

                    # 6. Course Title
                    course = "B.A. (Honours)"
                    course_match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', text_clean, re.IGNORECASE)
                    if course_match:
                        course = course_match.group(0).strip()

                    try:
                        admission_year = "20" + reg_no.split("-")[-1]
                    except:
                        admission_year = "Unknown"

                    # 7. Extract Semester Performance Table
                    sems_data = []
                    for line in text_clean.splitlines():
                        line_str = line.strip()
                        sem_m = re.search(r'^(I|II|III|IV|V|VI)\b\s*(\d{4})?\s*(\d+)?\s*(\d+)?\s*(\d+)?\s*([\d\.]+|N\.?A\.?)?', line_str)
                        if sem_m:
                            s_name = sem_m.group(1)
                            s_year = sem_m.group(2) or ""
                            s_marks = sem_m.group(4) or ""
                            s_sgpa = sem_m.group(6) or ""
                            sems_data.append((s_name, s_year, s_marks, s_sgpa))

                    # 8. Extract Final CGPA & Grade
                    overall_cgpa = "N.A."
                    overall_grade = "Fail / Semester Not Cleared"

                    cgpa_match = re.search(r'VI\b.*?\b([\d\.]+)\s+\d+\s+([\d\.]+)\s+([A-Z\+\-]+)', text_clean)
                    if cgpa_match:
                        overall_cgpa = cgpa_match.group(2)
                        overall_grade = cgpa_match.group(3)
                    else:
                        cgpa_sub = re.search(r'CGPA\s*[\:\s]*([\d\.]+)', text_clean, re.IGNORECASE)
                        if cgpa_sub: overall_cgpa = cgpa_sub.group(1)
                        grade_sub = re.search(r'Grade\s*[\:\s]*([A-Z\+\-]+)', text_clean, re.IGNORECASE)
                        if grade_sub: overall_grade = grade_sub.group(1)

                    print(f"Page {page_num+1} Extracted: Reg={reg_no}, Name={name}, Roll={roll_no}")

                    # 9. Database Upsert
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
                                registration_no=reg_no, semester=sem[0], year=sem[1], 
                                marks_obtained=sem[2], sgpa=sem[3]
                            ))
                    else:
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
                                registration_no=reg_no, semester=sem[0], year=sem[1], 
                                marks_obtained=sem[2], sgpa=sem[3]
                            ))
                            
                    extracted_count += 1
                except Exception as page_err:
                    print(f"Error on page {page_num+1}: {page_err}")
                    continue

        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")
    finally:
        if os.path.exists(temp_pdf_path): 
            os.remove(temp_pdf_path)

    return {"message": f"Successfully extracted {extracted_count} student record(s)!"}

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
            {"semester": s.semester, "year": s.year, "marks": s.marks_obtained, "sgpa": s.sgpa} 
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

import os
import re
import shutil
import secrets
import gc
from PIL import Image
import pytesseract
import pdfplumber

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
                    text = page.extract_text() or ""
                    
                    # Fallback to OCR if page has almost no text
                    if len(text.strip()) < 30:
                        try:
                            img = page.to_image(resolution=120).original
                            text = pytesseract.image_to_string(img, lang="eng+ben")
                            del img
                            gc.collect()
                        except Exception as ocr_err:
                            print(f"OCR Error on Page {page_num}: {ocr_err}")
                            continue

                    # 1. Registration Number Extraction (Exact CU Pattern XXX-XXXX-XXXX-XX)
                    reg_no = None
                    reg_match = re.search(r'Registration\s*No\.?\s*[\:\s]*([0-9]{3}\-[0-9]{4}\-[0-9]{4}\-[0-9]{2})', text, re.IGNORECASE)
                    if reg_match:
                        reg_no = reg_match.group(1).strip()
                    else:
                        fallback_reg = re.search(r'\b(\d{3}\-\d{4}\-\d{4}\-\d{2})\b', text)
                        if fallback_reg:
                            reg_no = fallback_reg.group(1).strip()

                    if not reg_no:
                        continue 

                    # 2. Roll Number Extraction (Exact CU Pattern XXXXXX-XX-XXXX)
                    roll_no = "Unknown"
                    roll_match = re.search(r'Roll\s*No\.?\s*[\:\s]*([0-9]{6}\-[0-9]{2}\-[0-9]{4})', text, re.IGNORECASE)
                    if roll_match:
                        roll_no = roll_match.group(1).strip()
                    else:
                        fallback_roll = re.search(r'\b(\d{6}\-\d{2}\-\d{4})\b', text)
                        if fallback_roll:
                            roll_no = fallback_roll.group(1).strip()

                    # 3. Student Name Extraction
                    name_match = re.search(r'Name\s*[\:\s]+([A-Za-z\s\.]+?)(?=Registration|Roll|\n|$)', text, re.IGNORECASE)
                    name = name_match.group(1).strip() if name_match else "Unknown Student"

                    # 4. Course / Exam Title Extraction
                    course_match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)\s+Semester[^\n\r,]+', text, re.IGNORECASE)
                    course = course_match.group(0).strip() if course_match else "B.A. (Honours)"
                    
                    try:
                        admission_year = "20" + reg_no.split("-")[-1]
                    except:
                        admission_year = "Unknown"

                    # 5. Extract Semester Table Rows directly from text lines
                    sem_pattern = re.compile(
                        r'^\s*(I|II|III|IV|V|VI)\s+(\d{4})\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d\.\sA-Za-z]+)', 
                        re.MULTILINE
                    )
                    
                    sems_data = []
                    for m in sem_pattern.finditer(text):
                        sem_name, yr, fm, mo, cred, sgpa = m.groups()
                        sems_data.append((sem_name, yr, mo, sgpa.strip()))

                    # 6. Overall CGPA and Grade Extraction (From Semester VI row)
                    overall_cgpa = "N.A."
                    overall_grade = "Fail / Not Cleared"
                    
                    cgpa_match = re.search(
                        r'VI\s+\d{4}\s+\d+\s+\d+\s+\d+\s+([\d\.]+)\s+\d+\s+([\d\.]+)\s+([A-Z\+\-]+)', 
                        text
                    )
                    if cgpa_match:
                        overall_cgpa = cgpa_match.group(2)
                        overall_grade = cgpa_match.group(3)

                    # 7. Save or Update Record in Database
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
                    print(f"Error processing page {page_num}: {page_err}")
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

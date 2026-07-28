from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
import os
import pdfplumber
import re
import shutil
import pytesseract
from PIL import Image
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# --- SECURITY (ADMIN LOGIN) ---
security = HTTPBasic()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "cuadmin123")

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401, detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# --- DATABASE SETUP (Cloud PostgreSQL / Local SQLite) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Student(Base):
    __tablename__ = "students"
    registration_no = Column(String, primary_key=True, index=True)
    roll_no = Column(String)
    name = Column(String)
    admission_year = Column(String)
    course = Column(String, default="Unknown Course") # BA MDC, BCOM Major, etc.
    overall_cgpa = Column(String) 
    overall_grade = Column(String)
    
    # NEW FIELDS: Checkboxes & Status
    marksheet_received = Column(Boolean, default=False)
    certificate_received = Column(Boolean, default=False)
    post_grad_status = Column(String, default="Unemployed") # Govt Job, Pvt Job, Business...
    post_grad_details = Column(String, default="") 
    proof_document_path = Column(String, default="") 
    
    semesters = relationship("SemesterRecord", back_populates="student")

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

# --- FASTAPI APP ---
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- WEB UI ROUTE (Secured with Login) ---
@app.get("/")
def serve_frontend(username: str = Depends(authenticate_admin)):
    return FileResponse("index.html")

# --- API ENDPOINTS ---
@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(file: UploadFile = File(...), db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    extracted_count = 0
    try:
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if len(text.strip()) < 50:
                    img = page.to_image(resolution=300).original
                    text = pytesseract.image_to_string(img)

                # Extract Basic Info
                name_match = re.search(r'Name[\s\.\:]+([A-Za-z\s]+)(?=\n|Registration|Roll)', text, re.IGNORECASE)
                reg_match = re.search(r'Registration\s*No[\.\:\s]*([A-Za-z0-9\-]+)', text, re.IGNORECASE)
                roll_match = re.search(r'Roll[\s\&]*No[\.\:\s]*([A-Za-z0-9\-]+)', text, re.IGNORECASE)
                course_match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)\s+(Major|MDC|Honours|General)', text, re.IGNORECASE)

                if not reg_match: continue 
                    
                reg_no = reg_match.group(1).strip()
                name = name_match.group(1).strip() if name_match else "Unknown"
                roll_no = roll_match.group(1).strip() if roll_match else "Unknown"
                course = course_match.group(0).strip().upper() if course_match else "Unknown Course"
                try: admission_year = "20" + reg_no.split("-")[-1]
                except: admission_year = "Unknown"

                # Extract Tables (Semesters, SGPA, CGPA)
                tables = page.extract_tables()
                sems_data = []
                overall_cgpa = "N.A."
                overall_grade = "Fail" # Default till proven otherwise
                
                if tables:
                    for table in tables:
                        if not table or len(table) < 2: continue
                        col1_row1 = str(table[1][0]).strip() if table[1] and len(table[1])>0 and table[1][0] else ""
                        
                        if 'I' in col1_row1 or 'Semester' in str(table[0]):
                            for row in table:
                                if not row or len(row) == 0 or not row[0]: continue
                                sem_name = str(row[0]).strip()
                                if sem_name in ['I', 'II', 'III', 'IV', 'V', 'VI']:
                                    yr = str(row[1]).strip() if len(row) > 1 and row[1] else ""
                                    mo = str(row[3]).strip() if len(row) > 3 and row[3] else ""
                                    sgpa = str(row[5]).strip() if len(row) > 5 and row[5] else ""
                                    sems_data.append((sem_name, yr, mo, sgpa))
                                    
                                    if sem_name == 'VI':
                                        if len(row) > 7 and row[7]: overall_cgpa = str(row[7]).strip()
                                        if len(row) > 8 and row[8]: overall_grade = str(row[8]).strip()

                # Save to Database
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if not existing_student:
                    student = Student(
                        registration_no=reg_no, roll_no=roll_no, name=name, 
                        admission_year=admission_year, course=course, 
                        overall_cgpa=overall_cgpa, overall_grade=overall_grade
                    )
                    db.add(student)
                    for sem in sems_data:
                        db.add(SemesterRecord(
                            registration_no=reg_no, semester=sem[0], year=sem[1], 
                            marks_obtained=sem[2], sgpa=sem[3]
                        ))
                    extracted_count += 1
        db.commit()
    finally:
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

    return {"message": f"Successfully extracted {extracted_count} full student records!"}

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
    if not student: raise HTTPException(status_code=404, detail="Student not found")

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

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found")
    
    sems = db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).all()
    
    # Try to deduce passout year from VI semester year
    passout_year = "Unknown"
    for s in sems:
        if s.semester == "VI" and s.year:
            passout_year = s.year

    return {
        "student": {
            "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
            "admission_year": student.admission_year, "passout_year": passout_year, "course": student.course,
            "cgpa": student.overall_cgpa, "grade": student.overall_grade,
            "marksheet_received": student.marksheet_received, "certificate_received": student.certificate_received,
            "status": student.post_grad_status, "details": student.post_grad_details, "proof": student.proof_document_path
        },
        "semesters": [{"semester": s.semester, "year": s.year, "marks": s.marks_obtained, "sgpa": s.sgpa} for s in sems]
    }

@app.get("/api/admin/grade-stats")
def get_grade_stats(db: Session = Depends(get_db), user: str = Depends(authenticate_admin)):
    # Group by Course and Grade, count how many students
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

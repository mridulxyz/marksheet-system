--- START OF FILE Paste August 01, 2026 - 9:43AM ---

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

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Boolean, func, text
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base

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

    raise HTTPException(
        status_code=401,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Basic"},
    )

def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin permissions required to perform this operation."
        )
    return user

# --- OPENAI CLIENT SETUP ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if (openai and OPENAI_API_KEY) else None

# --- DATABASE SETUP ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./calcutta_university.db")
if DATABASE_URL.startswith("postgres://"): 
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,
    pool_recycle=300,
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
    passout_year = Column(String, default="NIL") # Added Passout Year
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
    pdf_document_path = Column(String, default="")  # PDF Repository Path
    
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

# --- GLOBAL BACKGROUND PROGRESS TRACKER ---
bg_upload_status = {
    "is_processing": False,
    "total_pages": 0,
    "processed_pages": 0,
    "extracted_count": 0,
    "status_message": "Idle",
    "filename": ""
}

# --- OPENAI GPT-4o VISION PARSER ---
def parse_marksheet_with_openai_vision(page):
    pix = page.get_pixmap(dpi=200)
    img_bytes = pix.tobytes("jpeg")
    base64_image = base64.b64encode(img_bytes).decode('utf-8')

    prompt = """
    You are an expert transcript parser. Inspect this Calcutta University Grade Sheet image with 100% precision. Return ONLY a valid JSON object matching this schema:

    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)",
      "subject": "BNGA",
      "passout_year": "2025",
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified with Honours",
      "semesters": [
        {"semester": "I", "year": "2019", "full_marks": "400", "marks": "248", "credit": "20", "sgpa": "5.624"}
      ]
    }

    VERIFICATION RULES FOR 100% ACCURACY:
    1. PASSOUT YEAR: Extract passout year EXACTLY from the main heading at the very top of the page (e.g. "Examination (Under CBCS), 2025" means passout_year is "2025"). 
    2. STRICT FAIL LOGIC: Look at the Remarks section below the bottom summary table. If Remarks is exactly "Semester Cleared" (without the word Qualified) OR "Semester not cleared", this means the candidate FAILED or did not qualify overall.
    3. NO MARKS FOR FAILED CANDIDATES: If the candidate failed (based on rule 2), you MUST set "passout_year": "NIL", "overall_cgpa": "N.A.", "overall_grade": "Fail / Semester Not Cleared", and return an EMPTY ARRAY for "semesters": []. Do not extract marks for failed candidates!
    4. OVERALL GRADE/CGPA: Only read overall_grade/cgpa from row 'VI' in the bottom table. Never read text like 'Good' or 'P'.
    5. ONLY PRESENT SEMESTERS: If the student did qualify, read ONLY the bottom summary table for semesters. Do not skip any rows and do not read the top subject modules!
    6. SUBJECT CODE: Extract subject prefix like BNGA, ENGG, SANA etc.
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = ai_client.chat.completions.create(
                model="gpt-4o",
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
            return json.loads(response.choices[0].message.content)
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
        lines, current_line, last_y = [], [], None
        for w in sorted_words:
            x0, y0, word = w[0], w[1], w[4]
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
    if not text: return None
    t_fixed = re.sub(r'[—–_~]', '-', text).replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    match = re.search(r'(?:Registration|Regn|Reg|Registra)[^\d]{0,15}([0-9\-\s\.\/]{13,22})', t_fixed, re.IGNORECASE)
    if match:
        d = re.sub(r'\D', '', match.group(1))[:13]
        if len(d) >= 13: return f"{d[:3]}-{d[3:7]}-{d[7:11]}-{d[11:]}"
    all_d = re.findall(r'\b\d{13}\b', re.sub(r'\D', ' ', t_fixed))
    return f"{all_d[0][:3]}-{all_d[0][3:7]}-{all_d[0][7:11]}-{all_d[0][11:]}" if all_d else None

def extract_roll_no_bulletproof(text: str):
    t_fixed = re.sub(r'[—–_~]', '-', text or "").replace('O', '0').replace('I', '1').replace('l', '1').replace('S', '5')
    all_d = re.findall(r'\b\d{12}\b', re.sub(r'\D', ' ', t_fixed))
    return f"{all_d[0][:6]}-{all_d[0][6:8]}-{all_d[0][8:]}" if all_d else "Unknown"

def extract_name_bulletproof(text: str):
    match = re.search(r'Name\s*[\:\.]*\s*([A-Za-z\s\.]+?)(?=Registration|Regn|Roll|\d{3}\-|\n|$)', text or "", re.IGNORECASE)
    if match:
        raw = re.sub(r'[^A-Za-z\s\.]', '', match.group(1)).strip()
        if len(raw) > 2: return raw
    return "Unknown Student"

def parse_summary_table_local(table_text: str, full_text: str):
    sems, cgpa, grade, remarks, passout_year = [], "N.A.", "Fail / Semester Not Cleared", "Qualified", "NIL"
    if not table_text: return sems, cgpa, grade, remarks, passout_year

    # Extract year from header
    yr_match = re.search(r'Examination[\s\S]{0,80}?(\d{4})', full_text, re.IGNORECASE)
    if yr_match: passout_year = yr_match.group(1)

    text_clean = re.sub(r'(\d)\s*[\,\.]\s*(\d)', r'\1.\2', table_text)
    
    rem_match = re.search(r'Remarks\s*[\:\.]*\s*([^\n\r]+)', text_clean, re.IGNORECASE)
    if rem_match: remarks = rem_match.group(1).strip()

    # Fail logic: If remarks is specifically just "Semester Cleared" or "not cleared"
    is_failed = bool(re.search(r'(not\s*cleared)', remarks, re.IGNORECASE)) or remarks.strip().lower() == "semester cleared"
    
    if is_failed:
        return [], "N.A.", "Fail / Semester Not Cleared", remarks, "NIL"

    lines = [l.strip() for l in text_clean.splitlines() if l.strip()]
    for line in lines:
        for sem in ['I', 'II', 'III', 'IV', 'V', 'VI']:
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

    valid_gpas = [f for f in re.findall(r'\b([1-9]\.\d{2,3})\b', text_clean) if 1.0 <= float(f) <= 10.0]
    if valid_gpas: cgpa = valid_gpas[-1]
    grade_m = re.search(r'\b(A\+|B\+|C\+|A|B|C|D|O)(?!\w)', text_clean)
    if grade_m: grade = grade_m.group(1)

    return sems, cgpa, grade, remarks, passout_year

# --- BACKGROUND WORKER ---
def process_large_pdf_in_background(temp_pdf_path: str, selected_course: str):
    global bg_upload_status
    db = SessionLocal()
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        total_pages = len(doc)
        
        bg_upload_status.update({"is_processing": True, "total_pages": total_pages, "processed_pages": 0, "extracted_count": 0, "status_message": f"Processing page 1 of {total_pages}..."})
        ai_quota_exceeded, extracted_count = False, 0
        
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                reg_no = None
                normalized_semesters = []
                
                # STRATEGY 1: OPENAI VISION
                if ai_client and not ai_quota_exceeded:
                    try:
                        if page_num > 0: time.sleep(0.3)
                        data = parse_marksheet_with_openai_vision(page)
                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            bg_upload_status["processed_pages"] = page_num + 1
                            continue

                        roll_no = data.get("roll_no", "Unknown")
                        name = data.get("name", "Unknown Student")
                        course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "Unknown")
                        subject = data.get("subject", "BNGA")
                        remarks = data.get("remarks", "")
                        passout_year = str(data.get("passout_year", "NIL")).strip()
                        overall_cgpa = str(data.get("overall_cgpa", "N.A.")).strip()
                        overall_grade = str(data.get("overall_grade", "Fail / Semester Not Cleared")).strip()

                        # Enforce fail rule tightly regardless of hallucination
                        if overall_grade.lower() in ["none", "null", "n.a.", ""] or "not cleared" in remarks.lower() or remarks.strip().lower() == "semester cleared":
                            overall_cgpa = "N.A."
                            overall_grade = "Fail / Semester Not Cleared"
                            passout_year = "NIL"
                            normalized_semesters = []
                        else:
                            for sem in data.get("semesters", []):
                                if not isinstance(sem, dict): continue
                                raw_s = str(sem.get("semester") or "").strip().upper()
                                if not raw_s: continue
                                normalized_semesters.append({
                                    "semester": raw_s,
                                    "year": str(sem.get("year") or "-").strip(),
                                    "full_marks": str(sem.get("full_marks") or "400").strip(),
                                    "marks": str(sem.get("marks") or "-").strip(),
                                    "credit": str(sem.get("credit") or "20").strip(),
                                    "sgpa": str(sem.get("sgpa") or "N.A.").strip()
                                })
                    except Exception as ai_err:
                        if "429" in str(ai_err) or "rate" in str(ai_err) or "quota" in str(ai_err):
                            ai_quota_exceeded = True

                # STRATEGY 2: LOCAL FALLBACK
                if not ai_client or ai_quota_exceeded or not reg_no:
                    full_text = page.get_text("text") or ""
                    reg_no = extract_reg_no_bulletproof(full_text)
                    if not reg_no:
                        bg_upload_status["processed_pages"] = page_num + 1
                        continue

                    roll_no = extract_roll_no_bulletproof(full_text)
                    name = extract_name_bulletproof(full_text)
                    c_match = re.search(r'(B\.?A\.?|B\.?Sc\.?|B\.?Com\.?)[^\n\r,]*', full_text, re.IGNORECASE)
                    course = selected_course if (selected_course and selected_course != "AUTO") else (re.sub(r'Semester.*', '', c_match.group(0).strip(), flags=re.IGNORECASE) if c_match else "Unknown")
                    s_match = re.search(r'\b([A-Z]{3,4})\-[A-Z0-9]+\b', full_text)
                    subject = s_match.group(1).upper() if s_match else "BNGA"

                    table_text = extract_text_rows_from_rect(page, fitz.Rect(0, page.rect.height * 0.48, page.rect.width, page.rect.height * 0.88))
                    normalized_semesters, overall_cgpa, overall_grade, remarks, passout_year = parse_summary_table_local(table_text, full_text)

                pdf_repo_path = f"uploads/pdf_repository/{reg_no}.pdf"
                try:
                    new_pdf = fitz.open()
                    new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)
                    new_pdf.save(pdf_repo_path)
                    new_pdf.close()
                except Exception: pdf_repo_path = ""

                # Upsert to DB
                existing = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing:
                    existing.name, existing.roll_no = name, roll_no
                    existing.course, existing.subject = course, subject
                    existing.overall_cgpa, existing.overall_grade, existing.remarks, existing.passout_year = overall_cgpa, overall_grade, remarks, passout_year
                    if pdf_repo_path: existing.pdf_document_path = pdf_repo_path
                    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
                else:
                    adm = "20" + reg_no.split("-")[-1] if "-" in reg_no else "Unknown"
                    existing = Student(registration_no=reg_no, roll_no=roll_no, name=name, admission_year=adm, course=course, subject=subject, passout_year=passout_year, overall_cgpa=overall_cgpa, overall_grade=overall_grade, remarks=remarks, pdf_document_path=pdf_repo_path)
                    db.add(existing)

                for sem in normalized_semesters:
                    db.add(SemesterRecord(registration_no=reg_no, semester=sem["semester"], year=sem["year"], full_marks=sem["full_marks"], marks_obtained=sem["marks"], credit=sem["credit"], sgpa=sem["sgpa"]))

                extracted_count += 1
                bg_upload_status.update({"processed_pages": page_num + 1, "extracted_count": extracted_count, "status_message": f"Processing page {page_num+1} of {total_pages} ({extracted_count} records saved)..."})
                if page_num % 5 == 0: db.commit(); gc.collect()

            except Exception:
                db.rollback()
                bg_upload_status["processed_pages"] = page_num + 1
                continue

        if doc: doc.close()
        db.commit()
        bg_upload_status.update({"is_processing": False, "status_message": f"🎉 ✅ EXTRACTION DONE! {extracted_count} records saved."})
    except Exception as e:
        db.rollback()
        bg_upload_status.update({"is_processing": False, "status_message": f"❌ Error: {str(e)}"})
    finally:
        db.close()
        if os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)

# --- FASTAPI APP SETUP ---
app = FastAPI(title="Shyampur Siddheswari Mahavidyalaya Portal")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

@app.on_event("startup")
def startup_db_setup():
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        try: conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS passout_year VARCHAR DEFAULT 'NIL';"))
        except: pass
        try: conn.execute(text("ALTER TABLE students ADD COLUMN passout_year VARCHAR DEFAULT 'NIL';"))
        except: pass

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.get("/api/auth/me")
def get_auth_me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(background_tasks: BackgroundTasks, file: UploadFile = File(...), selected_course: str = Form("AUTO"), user: dict = Depends(require_admin)):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()
    background_tasks.add_task(process_large_pdf_in_background, temp_pdf_path, selected_course)
    return {"message": f"🚀 Started background extraction for {total_pages} page(s)."}

@app.get("/api/admin/upload-status")
def get_upload_status(user: dict = Depends(get_current_user)):
    return bg_upload_status

@app.get("/api/admin/dropdown-options")
def get_dropdown_options(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    courses = [r[0] for r in db.query(Student.course).distinct().all() if r[0]]
    subjects = [r[0] for r in db.query(Student.subject).distinct().all() if r[0]]
    years = [r[0] for r in db.query(Student.passout_year).distinct().all() if r[0] and r[0] != "NIL"]
    return {"courses": sorted(list(set(courses))), "subjects": sorted(list(set(subjects))), "years": sorted(list(set(years)), reverse=True)}

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404, detail="Student not found.")
    sems = db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).all()
    return {
        "student": {
            "name": student.name, "reg_no": student.registration_no, "roll_no": student.roll_no,
            "admission_year": student.admission_year, "passout_year": student.passout_year, 
            "course": student.course, "subject": student.subject or "BNGA",
            "cgpa": student.overall_cgpa, "grade": student.overall_grade, "remarks": student.remarks,
            "marksheet_received": student.marksheet_received, "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date or "", "certificate_issue_date": student.certificate_issue_date or "",
            "issued_by": student.issued_by or "", "status": student.post_grad_status or "Unknown", 
            "details": student.post_grad_details, "proof": student.proof_document_path, "pdf_path": student.pdf_document_path
        },
        "semesters": [{"semester": s.semester, "year": s.year, "full_marks": s.full_marks, "marks": s.marks_obtained, "credit": s.credit, "sgpa": s.sgpa} for s in sems]
    }

@app.post("/api/admin/update-profile-full/{reg_no}")
async def update_student_profile_full(reg_no: str, payload: dict, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404)
    student.name, student.roll_no = payload.get("name", student.name), payload.get("roll_no", student.roll_no)
    student.course, student.subject = payload.get("course", student.course), payload.get("subject", student.subject)
    student.overall_cgpa, student.overall_grade = payload.get("cgpa", student.overall_cgpa), payload.get("grade", student.overall_grade)
    student.remarks = payload.get("remarks", student.remarks)

    if payload.get("semesters"):
        db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
        for sem in payload.get("semesters", []):
            db.add(SemesterRecord(registration_no=reg_no, semester=sem.get("semester"), year=sem.get("year"), full_marks=sem.get("full_marks"), marks_obtained=sem.get("marks"), credit=sem.get("credit"), sgpa=sem.get("sgpa")))
    db.commit()
    return {"message": "Updated!"}

@app.post("/api/admin/update-issuance-detailed/{reg_no}")
async def update_issuance_detailed(reg_no: str, marksheet_received: bool=Form(...), certificate_received: bool=Form(...), marksheet_issue_date: str=Form(""), certificate_issue_date: str=Form(""), issued_by: str=Form(""), db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    student.marksheet_received, student.certificate_received = marksheet_received, certificate_received
    student.marksheet_issue_date, student.certificate_issue_date = marksheet_issue_date, certificate_issue_date
    student.issued_by = issued_by
    db.commit()
    return {"message": "Updated!"}

@app.post("/api/admin/update-status/{reg_no}")
async def update_student_status(reg_no: str, course: str=Form(...), marksheet_received: bool=Form(...), certificate_received: bool=Form(...), status: str=Form(...), details: str=Form(""), proof_file: UploadFile=File(None), db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    student.course, student.post_grad_status, student.post_grad_details = course, status, details
    if proof_file:
        file_path = os.path.join("uploads", f"{reg_no}_proof.{proof_file.filename.split('.')[-1]}")
        with open(file_path, "wb") as buffer: shutil.copyfileobj(proof_file.file, buffer)
        student.proof_document_path = file_path
    db.commit()
    return {"message": "Updated!"}

@app.get("/api/admin/all-students")
def get_all_students(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    return db.query(Student).all()

@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: raise HTTPException(status_code=404)
    if student.pdf_document_path and os.path.exists(student.pdf_document_path):
        try: os.remove(student.pdf_document_path)
        except: pass
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    db.delete(student)
    db.commit()
    return {"message": "Deleted"}

@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    db.query(SemesterRecord).delete()
    db.query(Student).delete()
    db.commit()
    try:
        shutil.rmtree("uploads/pdf_repository")
        os.makedirs("uploads/pdf_repository", exist_ok=True)
    except: pass
    return {"message": "Cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)

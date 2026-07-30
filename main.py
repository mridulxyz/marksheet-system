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
            detail="Admin permissions required to perform delete operations."
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

# --- GLOBAL BACKGROUND PROGRESS TRACKER ---
bg_upload_status = {
    "is_processing": False,
    "total_pages": 0,
    "processed_pages": 0,
    "extracted_count": 0,
    "status_message": "Idle",
    "filename": ""
}

# --- CROP ANCHOR SEARCH ---

def get_summary_table_crop_rect(page):
    rect = page.rect
    hits = page.search_for("Cumulative Credit") or page.search_for("Semester Credit") or page.search_for("Full Marks")
    
    if hits:
        summary_hits = [h for h in hits if h.y0 > rect.height * 0.45]
        if summary_hits:
            y0 = max(0, summary_hits[0].y0 - 20)
            y1 = min(rect.height, y0 + (rect.height * 0.35))
            return fitz.Rect(0, y0, rect.width, y1)

    return fitz.Rect(0, rect.height * 0.48, rect.width, rect.height * 0.88)

# --- FLAGSHIP GPT-4o VISION PARSER ---

def parse_marksheet_with_openai_vision(page):
    rect = page.rect

    # 1. Full Page Image for Header Info
    pix_full = page.get_pixmap(dpi=120)
    img_bytes_full = pix_full.tobytes("jpeg")
    base64_full = base64.b64encode(img_bytes_full).decode('utf-8')

    prompt_header = """
    Extract header information from this Calcutta University Marksheet image and return ONLY a JSON object:
    {
      "registration_no": "424-1211-0240-19",
      "roll_no": "192424-11-0044",
      "name": "SWAGATA PURKAIT",
      "course": "B.A. (Honours) Examination (Under CBCS)"
    }
    Rules:
    1. Registration No format: 3-4-4-2 (e.g. 424-1211-0240-19).
    2. Roll No format: 6-2-4 (e.g. 192424-11-0044).
    3. Omit 'Semester - VI' or 'Semester - I' from course title.
    """

    # 2. Cropped Summary Table Image (CROPS OUT TOP SUBJECT TABLES)
    table_rect = get_summary_table_crop_rect(page)
    pix_table = page.get_pixmap(dpi=200, clip=table_rect)
    img_bytes_table = pix_table.tobytes("jpeg")
    base64_table = base64.b64encode(img_bytes_table).decode('utf-8')

    prompt_table = """
    Analyze ONLY this cropped Semester Summary Table from the bottom of a Calcutta University Grade Sheet and return ONLY a JSON object:

    {
      "overall_cgpa": "6.819",
      "overall_grade": "B+",
      "remarks": "Qualified with Honours",
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
    1. EXTRACT ALL 6 ROWS: Carefully read every row present in the summary table (I, II, III, IV, V, VI). Do NOT omit Semester IV!
    2. READ VALUES ROW BY ROW:
       - Column 1: Semester (I, II, III, IV, V, VI)
       - Column 2: Year (e.g. 2019, 2020, 2022, 2023, 2024)
       - Column 3: Full Marks (400 or 500)
       - Column 4: Marks Obtained (exact number e.g. 248, 310, 330, 372, 263, 270)
       - Column 5: Semester Credit (20, 24, 26)
       - Column 6: SGPA (e.g. 5.624, 7.705, 6.899, 7.367, 6.527, 6.686)
    3. OVERALL GRADE: Read 'overall_grade' ONLY from column 'Letter Grade' in row 'VI' (e.g. B+, A+, A, B, C+, C, D, O). Do NOT read words like 'Good' or status 'P'.
    4. OVERALL CGPA: Read 'overall_cgpa' ONLY from column 'CGPA' in row 'VI' (e.g. 6.819).
    5. IF SEMESTER NOT CLEARED: If 'Semester not cleared' is printed in Remarks, set 'overall_cgpa': 'N.A.', 'overall_grade': 'Fail / Semester Not Cleared', and 'semesters': [].
    6. REMARKS: Read the exact Remarks line right below the table (e.g. 'Qualified with Honours' or 'Semester not cleared').
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_table = ai_client.chat.completions.create(
                model="gpt-4o",
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
                model="gpt-4o",
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

            extracted_grade = str(table_data.get("overall_grade", "Fail / Semester Not Cleared")).strip()
            valid_cu_grades = ["O", "A+", "A", "B+", "B", "C+", "C", "D", "F", "FAIL / SEMESTER NOT CLEARED"]
            
            if extracted_grade.upper() not in valid_cu_grades:
                grade_m = re.search(r'\b(A\+|B\+|C\+|A|B|C|D|O|F)(?!\w)', extracted_grade, re.IGNORECASE)
                if grade_m:
                    extracted_grade = grade_m.group(1).upper()

            return {
                "registration_no": header_data.get("registration_no"),
                "roll_no": header_data.get("roll_no"),
                "name": header_data.get("name"),
                "course": header_data.get("course"),
                "overall_cgpa": table_data.get("overall_cgpa", "N.A."),
                "overall_grade": extracted_grade,
                "remarks": table_data.get("remarks", "Qualified with Honours"),
                "semesters": table_data.get("semesters", [])
            }
        except Exception as e:
            err_msg = str(e).lower()
            if ("429" in err_msg or "rate" in err_msg or "quota" in err_msg) and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 3)
            else:
                raise e

# --- LOCAL FALLBACK PARSER (WITH 1D Y-OVERLAP CLUSTERING) ---

def group_words_into_horizontal_lines(words, y_threshold=8):
    if not words: return []
    words_sorted = sorted(words, key=lambda w: w[1])

    lines = []
    for w in words_sorted:
        placed = False
        y0, y1 = w[1], w[3]
        y_center = (y0 + y1) / 2.0

        for line in lines:
            line_y_avg = sum((item[1] + item[3]) / 2.0 for item in line) / len(line)
            if abs(y_center - line_y_avg) <= y_threshold:
                line.append(w)
                placed = True
                break

        if not placed:
            lines.append([w])

    formatted_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w[0])
        line_text = " ".join(w[4] for w in line_sorted)
        formatted_lines.append(line_text)

    return formatted_lines

def extract_summary_table_text_pymupdf(page):
    table_rect = get_summary_table_crop_rect(page)
    words = page.get_text("words", clip=table_rect)
    if not words: return page.get_text("text", clip=table_rect) or ""
    lines = group_words_into_horizontal_lines(words)
    return "\n".join(lines)

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

    if is_not_cleared:
        return [], "N.A.", "Fail / Semester Not Cleared", remarks

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
                    s_fm = integers[0] if int(integers[0]) in [400, 500] else "400"
                    s_marks = integers[1] if (len(integers) >= 2 and integers[0] in ["400", "500"]) else integers[0]
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

# --- BACKGROUND WORKER FOR 1000-PAGE PDFs ---

def process_large_pdf_in_background(temp_pdf_path: str, selected_course: str):
    global bg_upload_status
    db = SessionLocal()
    
    try:
        doc = fitz.open(temp_pdf_path) if fitz else []
        total_pages = len(doc)
        
        bg_upload_status["is_processing"] = True
        bg_upload_status["total_pages"] = total_pages
        bg_upload_status["processed_pages"] = 0
        bg_upload_status["extracted_count"] = 0
        bg_upload_status["status_message"] = f"Processing page 1 of {total_pages}..."
        
        ai_quota_exceeded = False
        extracted_count = 0
        
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                reg_no = None
                normalized_semesters = []
                remarks = "Qualified"
                subject = "BNGA"
                
                # --- STRATEGY 1: OPENAI VISION (GPT-4o) ---
                if ai_client and not ai_quota_exceeded:
                    try:
                        if page_num > 0:
                            time.sleep(0.3)

                        data = parse_marksheet_with_openai_vision(page)

                        reg_no = data.get("registration_no")
                        if not reg_no or reg_no == "null":
                            bg_upload_status["processed_pages"] = page_num + 1
                            continue

                        roll_no = data.get("roll_no", "Unknown")
                        name = data.get("name", "Unknown Student")
                        course = selected_course if (selected_course and selected_course != "AUTO") else data.get("course", "B.A. (Honours) Examination (Under CBCS)")
                        subject = data.get("subject", "BNGA")
                        
                        remarks = data.get("remarks", "Qualified with Honours")
                        overall_cgpa = data.get("overall_cgpa", "N.A.")
                        overall_grade = data.get("overall_grade", "Fail / Semester Not Cleared")

                        # FOR FAILED / UNCLEARED CANDIDATES: DO NOT STORE SEMESTERS!
                        if "not cleared" in str(remarks).lower() or "fail" in str(overall_grade).lower():
                            overall_cgpa = "N.A."
                            overall_grade = "Fail / Semester Not Cleared"
                            normalized_semesters = []
                        else:
                            raw_semesters = data.get("semesters", [])
                            for sem in raw_semesters:
                                if not isinstance(sem, dict): continue
                                raw_s = str(sem.get("semester") or sem.get("sem") or "").strip().upper()
                                raw_s = re.sub(r'SEMESTER\s*', '', raw_s).strip()
                                if not raw_s: continue

                                yr_val = str(sem.get("year") or sem.get("exam_year") or "").strip()
                                sgpa_val = str(sem.get("sgpa") or sem.get("gpa") or "").strip()
                                marks_val = str(sem.get("marks") or sem.get("marks_obtained") or "").strip()

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
                        bg_upload_status["processed_pages"] = page_num + 1
                        continue

                    roll_no = extract_roll_no_bulletproof(full_text)
                    name = extract_name_bulletproof(full_text)
                    course = selected_course if (selected_course and selected_course != "AUTO") else extract_course(full_text)

                    subj_match = re.search(r'\b([A-Z]{3,4})\-[A-Z0-9]+\b', full_text)
                    subject = subj_match.group(1).upper() if subj_match else "BNGA"

                    table_text = extract_summary_table_text_pymupdf(page)
                    normalized_semesters, overall_cgpa, overall_grade, remarks = parse_summary_table_local(table_text)

                # Database Upsert
                existing_student = db.query(Student).filter(Student.registration_no == reg_no).first()
                if existing_student:
                    existing_student.name = name
                    existing_student.roll_no = roll_no
                    existing_student.course = course
                    existing_student.subject = subject
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
                        subject=subject,
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

                extracted_count += 1
                bg_upload_status["processed_pages"] = page_num + 1
                bg_upload_status["extracted_count"] = extracted_count
                bg_upload_status["status_message"] = f"Processing page {page_num+1} of {total_pages} ({extracted_count} records saved so far)..."

                if page_num % 5 == 0:
                    db.commit()
                    gc.collect()

            except Exception as page_err:
                db.rollback()
                bg_upload_status["processed_pages"] = page_num + 1
                continue

        if doc: doc.close()
        db.commit()
        
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"🎉 ✅ EXTRACTION DONE! Successfully extracted {extracted_count} student record(s) from {total_pages} pages into the database."
    except Exception as e:
        db.rollback()
        bg_upload_status["is_processing"] = False
        bg_upload_status["status_message"] = f"❌ Error during background extraction: {str(e)}"
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
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS subject VARCHAR DEFAULT 'BNGA';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS remarks VARCHAR DEFAULT 'Qualified';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_received BOOLEAN DEFAULT FALSE;"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS marksheet_issue_date VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS certificate_issue_date VARCHAR DEFAULT '';"))
                conn.execute(text("ALTER TABLE students ADD COLUMN IF NOT EXISTS issued_by VARCHAR DEFAULT '';"))
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
                    if "subject" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN subject VARCHAR DEFAULT 'BNGA';"))
                    if "remarks" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN remarks VARCHAR DEFAULT 'Qualified';"))
                    if "marksheet_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_received BOOLEAN DEFAULT 0;"))
                    if "certificate_received" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_received BOOLEAN DEFAULT 0;"))
                    if "marksheet_issue_date" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN marksheet_issue_date VARCHAR;"))
                    if "certificate_issue_date" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN certificate_issue_date VARCHAR;"))
                    if "issued_by" not in columns: conn.execute(text("ALTER TABLE students ADD COLUMN issued_by VARCHAR;"))
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
def serve_frontend(user: dict = Depends(get_current_user)):
    return FileResponse("index.html")

# --- AUTH INFO ENDPOINT ---
@app.get("/api/auth/me")
def get_auth_me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}

# --- API ENDPOINTS ---

@app.get("/api/admin/upload-status")
def get_upload_status(user: dict = Depends(get_current_user)):
    return bg_upload_status

@app.post("/api/admin/upload-marksheet")
async def upload_marksheet(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    selected_course: str = Form("AUTO"),
    user: dict = Depends(get_current_user)
):
    temp_pdf_path = f"temp_{secrets.token_hex(4)}_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc = fitz.open(temp_pdf_path) if fitz else []
    total_pages = len(doc)
    doc.close()

    background_tasks.add_task(process_large_pdf_in_background, temp_pdf_path, selected_course)

    return {
        "message": f"🚀 Successfully started background processing for {total_pages} page(s)! Refresh Tab 6 (Student Directory) to see extracted records in real-time."
    }

@app.get("/api/student/{reg_no}")
def get_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
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
            "subject": student.subject or "BNGA",
            "cgpa": student.overall_cgpa, 
            "grade": student.overall_grade,
            "remarks": student.remarks or "Qualified",
            "marksheet_received": student.marksheet_received, 
            "certificate_received": student.certificate_received,
            "marksheet_issue_date": student.marksheet_issue_date or "",
            "certificate_issue_date": student.certificate_issue_date or "",
            "issued_by": student.issued_by or "",
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

# Full Manual Profile & Marks Update (Menu 2 Edit Provision)
@app.post("/api/admin/update-profile-full/{reg_no}")
async def update_student_profile_full(
    reg_no: str,
    payload: dict,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")

    student.name = payload.get("name", student.name)
    student.roll_no = payload.get("roll_no", student.roll_no)
    student.course = payload.get("course", student.course)
    student.subject = payload.get("subject", student.subject)
    student.overall_cgpa = payload.get("cgpa", student.overall_cgpa)
    student.overall_grade = payload.get("grade", student.overall_grade)
    student.remarks = payload.get("remarks", student.remarks)

    # Update Semesters
    new_semesters = payload.get("semesters", [])
    if new_semesters:
        db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
        for sem in new_semesters:
            db.add(SemesterRecord(
                registration_no=reg_no,
                semester=sem.get("semester"),
                year=sem.get("year"),
                full_marks=sem.get("full_marks", "400"),
                marks_obtained=sem.get("marks"),
                credit=sem.get("credit", "20"),
                sgpa=sem.get("sgpa")
            ))

    db.commit()
    return {"message": "Student profile and marks updated successfully!"}

# Detailed Document Issuance Update (Menu 3)
@app.post("/api/admin/update-issuance-detailed/{reg_no}")
async def update_issuance_detailed(
    reg_no: str,
    marksheet_received: bool = Form(...),
    certificate_received: bool = Form(...),
    marksheet_issue_date: str = Form(""),
    certificate_issue_date: str = Form(""),
    issued_by: str = Form(""),
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user)
):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")

    student.marksheet_received = marksheet_received
    student.certificate_received = certificate_received
    student.marksheet_issue_date = marksheet_issue_date
    student.certificate_issue_date = certificate_issue_date
    student.issued_by = issued_by

    db.commit()
    return {"message": "Issuance details updated successfully!"}

# ADMIN ONLY: Delete Single Student
@app.delete("/api/admin/student/{reg_no}")
def delete_student(reg_no: str, db: Session = Depends(get_db), user: dict = Depends(require_admin)):
    student = db.query(Student).filter(Student.registration_no == reg_no).first()
    if not student: 
        raise HTTPException(status_code=404, detail="Student not found")
    
    db.query(SemesterRecord).filter(SemesterRecord.registration_no == reg_no).delete()
    db.delete(student)
    db.commit()
    return {"message": f"Student {reg_no} deleted successfully"}

# ADMIN ONLY: Clear All Database Records
@app.post("/api/admin/clear-all-students")
def clear_all_students(db: Session = Depends(get_db), user: dict = Depends(require_admin)):
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
    user: dict = Depends(get_current_user)
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
def get_grade_stats(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
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
def get_all_students(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    return db.query(Student).all()

# --- ENTRY POINT FOR UVICORN ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

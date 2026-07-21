from sqlalchemy import create_engine, Column, Integer, String, Float, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, sessionmaker
import datetime

DATABASE_URL = "sqlite:///./ocr_analysis.db"

engine = create_engine(
    DATABASE_URL,
    connect_args=
    {"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase): pass

class Run(Base):
    __tablename__ ="runs"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="running")
    progress = Column(String, default="")
    error = Column(Text, nullable=True)

class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"), nullable=True)
    image_name = Column(String, index=True)
    doc_type = Column(String, index=True)
    engine = Column(String, index=True)
    model_name = Column(String, index=True)
    total_time_seconds = Column(Float)
    avg_confidence = Column(Float)
    cer = Column(Float,nullable=True)
    wer = Column(Float, nullable=True)
    common_field_match_ratio = Column(Float, nullable=True)
    raw_text = Column(Text)
    words = Column(JSON)
    common_field_results = Column(JSON)
    settings_used = Column(JSON)
    preprocessing_used = Column(JSON)
    image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
from db import Base
from sqlalchemy import Column, Integer, String, Boolean
from pgvector.sqlalchemy import Vector

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    content = Column(String)
    domain = Column(String, index=True)
    verified = Column(Boolean, index=True)
    year = Column(Integer)
    embedding = Column(Vector(384))  # matches your vector(384) column in DB
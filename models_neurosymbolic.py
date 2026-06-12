from db import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from pgvector.sqlalchemy import Vector

class KGNode(Base):
    __tablename__ = "kg_nodes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    category = Column(String, index=True) # e.g. symptom, diagnosis, risk_factor, intervention
    severity = Column(Float, default=0.0) # r_i severity metric
    embedding = Column(Vector(384)) # matching MiniLM embedding dimension

class KGEdge(Base):
    __tablename__ = "kg_edges"

    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False)
    target_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False)
    relationship = Column(String, nullable=False) # e.g. increases_risk_for, maintains, requires
    weight = Column(Float, default=1.0) # w_ij connection weight

class DocumentKGMapping(Base):
    __tablename__ = "document_kg_mappings"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    node_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="CASCADE"), nullable=False)

class Workflow(Base):
    __tablename__ = "procedural_workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)

class WorkflowStep(Base):
    __tablename__ = "workflow_steps"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("procedural_workflows.id", ondelete="CASCADE"), nullable=False)
    sequence = Column(Integer, nullable=False, index=True) # order of the step
    title = Column(String, nullable=False)
    concept_id = Column(Integer, ForeignKey("kg_nodes.id", ondelete="SET NULL"), nullable=True) # maps step to clinical entity

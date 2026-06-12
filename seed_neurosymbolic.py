import os
import sys
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import SessionLocal, engine
from models import Document
from models_neurosymbolic import KGNode, KGEdge, DocumentKGMapping, Workflow, WorkflowStep
from sentence_transformers import SentenceTransformer

# Load sentence transformer model to generate node and document embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def seed_data():
    db = SessionLocal()
    try:
        # Clear existing data to allow re-seeding
        db.query(DocumentKGMapping).delete()
        db.query(KGEdge).delete()
        db.query(KGNode).delete()
        db.query(WorkflowStep).delete()
        db.query(Workflow).delete()
        
        # Clean up previously seeded clinical documents to avoid duplicates
        clinical_titles = [
            "Clinical Guideline: Initial Screening for Major Depression",
            "Clinical Practice Standard: Patient Consultation Methods",
            "Columbia-Suicide Severity Rating Scale (C-SSRS) Risk Assessment",
            "DSM-5 Structural Diagnostic Interview Protocols",
            "Cognitive Behavioral Therapy (CBT) and Trauma-Informed Interventions"
        ]
        db.query(Document).filter(Document.title.in_(clinical_titles)).delete()
        
        # Clean up DevOps release documents
        devops_titles = [
            "DevOps Guide: CI Build Pipeline Standards",
            "QA Practice: Automated Test Suite Implementation",
            "Security Standard: Static & Dynamic Vulnerability Scanning",
            "Release Practice: Staging Validation & Verification Protocols",
            "Deployment standard: Canary Releases and Production Rollout"
        ]
        db.query(Document).filter(Document.title.in_(devops_titles)).delete()
        db.commit()

        print("Cleared existing neurosymbolic tables...")

        # 1. Define KG Nodes (Clinical & DevOps)
        nodes_data = [
            # ── Clinical Symptoms & Protocols ──
            {"name": "depressed_mood", "description": "Persistent feelings of sadness, emptiness, or low mood.", "category": "symptom", "severity": 0.5},
            {"name": "anhedonia", "description": "Markedly diminished interest or pleasure in all or almost all activities.", "category": "symptom", "severity": 0.6},
            {"name": "chronic_insomnia", "description": "Difficulty falling or staying asleep, leading to sleep deprivation.", "category": "symptom", "severity": 0.4},
            {"name": "rumination", "description": "Obsessive thinking about negative events or experiences.", "category": "symptom", "severity": 0.3},
            {"name": "ACE_exposure", "description": "Adverse Childhood Experiences (ACE) exposure.", "category": "risk_factor", "severity": 0.8},
            {"name": "childhood_abuse", "description": "Physical, emotional, or psychological abuse during childhood.", "category": "risk_factor", "severity": 0.9},
            {"name": "depression_screening", "description": "Initial screening tools to detect potential depressive mood concerns.", "category": "protocol", "severity": 0.0},
            {"name": "clinical_consultation", "description": "Clinical intake and consultation to gain situational context.", "category": "protocol", "severity": 0.0},
            {"name": "risk_assessment", "description": "Assessment of psychological risk factors, self-harm, and suicidal ideation.", "category": "protocol", "severity": 0.0},
            {"name": "diagnostic_interview", "description": "Structured clinical interview applying formal diagnostic criteria.", "category": "protocol", "severity": 0.0},
            {"name": "trauma_informed_treatment", "description": "Guided interventions, cognitive behavioral therapy, and trauma recovery protocols.", "category": "intervention", "severity": 0.0},
            
            # ── DevOps Release pipeline concepts ──
            {"name": "ci_build", "description": "Continuous Integration automated compilation and artifact build process.", "category": "devops", "severity": 0.1},
            {"name": "automated_testing", "description": "Execution of automated unit, integration, and API testing suites.", "category": "devops", "severity": 0.2},
            {"name": "vulnerability_scan", "description": "Static and dynamic analysis scan (SAST/DAST) for security issues.", "category": "security", "severity": 0.5},
            {"name": "staging_environment", "description": "QA validation and verification in a replica staging environment.", "category": "devops", "severity": 0.3},
            {"name": "canary_release", "description": "Gradual release rollout to production and telemetry validation.", "category": "devops", "severity": 0.4}
        ]

        print("Generating embeddings for KG nodes...")
        inserted_nodes = {}
        for n in nodes_data:
            # Generate embedding vector
            text_to_embed = f"{n['name']}: {n['description']}"
            embedding = model.encode([text_to_embed])[0].tolist()
            
            node = KGNode(
                name=n["name"],
                description=n["description"],
                category=n["category"],
                severity=n["severity"],
                embedding=embedding
            )
            db.add(node)
            db.flush() # flush to get node.id
            inserted_nodes[n["name"]] = node

        db.commit()
        print(f"Inserted {len(inserted_nodes)} KG nodes.")

        # 2. Define KG Edges
        edges_data = [
            # Clinical edges
            ("childhood_abuse", "ACE_exposure", "increases_risk_for", 0.9),
            ("ACE_exposure", "chronic_insomnia", "increases_risk_for", 0.7),
            ("chronic_insomnia", "depressed_mood", "maintains", 0.6),
            ("depressed_mood", "anhedonia", "maintains", 0.5),
            ("anhedonia", "trauma_informed_treatment", "requires", 0.8),
            
            # DevOps edges
            ("ci_build", "automated_testing", "precedes", 0.9),
            ("automated_testing", "vulnerability_scan", "precedes", 0.8),
            ("vulnerability_scan", "staging_environment", "precedes", 0.7),
            ("staging_environment", "canary_release", "precedes", 0.9)
        ]

        for src, tgt, rel, weight in edges_data:
            edge = KGEdge(
                source_id=inserted_nodes[src].id,
                target_id=inserted_nodes[tgt].id,
                relationship=rel,
                weight=weight
            )
            db.add(edge)
        db.commit()
        print(f"Inserted {len(edges_data)} relationships/edges.")

        # 3. Create Workflows
        # Workflow A: PHQ-9 Clinical Workflow
        workflow_phq = Workflow(
            name="PHQ-9",
            description="Clinical step-by-step screening, assessment, and treatment workflow for Depression."
        )
        db.add(workflow_phq)
        db.flush()

        steps_phq = [
            (1, "Initial Screening (Depression)", "depression_screening"),
            (2, "Clinical Consultation", "clinical_consultation"),
            (3, "Risk Assessment", "risk_assessment"),
            (4, "Diagnostic Interview", "diagnostic_interview"),
            (5, "Guided Intervention (CBT)", "trauma_informed_treatment")
        ]

        for seq, title, concept_name in steps_phq:
            step = WorkflowStep(
                workflow_id=workflow_phq.id,
                sequence=seq,
                title=title,
                concept_id=inserted_nodes[concept_name].id
            )
            db.add(step)
            
        # Workflow B: Software Release DevOps Workflow
        workflow_release = Workflow(
            name="Software_Release",
            description="Automated CI/CD DevOps pipeline validation and canary rollout steps."
        )
        db.add(workflow_release)
        db.flush()

        steps_release = [
            (1, "CI Compilation & Artifact Build", "ci_build"),
            (2, "Automated Testing Suite", "automated_testing"),
            (3, "Security Vulnerability Scan", "vulnerability_scan"),
            (4, "Staging Environment Verification", "staging_environment"),
            (5, "Production Canary Rollout", "canary_release")
        ]

        for seq, title, concept_name in steps_release:
            step = WorkflowStep(
                workflow_id=workflow_release.id,
                sequence=seq,
                title=title,
                concept_id=inserted_nodes[concept_name].id
            )
            db.add(step)

        db.commit()
        print("Created PHQ-9 and Software_Release workflows and steps.")

        # 4. Insert Clinical & DevOps Guidelines Documents
        docs_data = [
            # ── Clinical Guidelines ──
            {
                "title": "Clinical Guideline: Initial Screening for Major Depression",
                "content": "GUIDELINE SECTION 1: INITIAL SCREENING. Systematic depression screening tools should trigger deeper assessment when needed. Screen patients reporting low mood, sadness, or lack of pleasure (anhedonia) using standardized instruments like PHQ-2 or PHQ-9. Positive screens flag a potential concern and must lead directly to a formal clinical consultation.",
                "domain": "health",
                "verified": True,
                "year": 2025,
                "concept": "depression_screening"
            },
            {
                "title": "Clinical Practice Standard: Patient Consultation Methods",
                "content": "GUIDELINE SECTION 2: CLINICAL CONSULTATION. Intake and consultation. After a positive screening result, conduct an initial clinical consultation. The provider should obtain context about the duration of symptoms, past history, and situational stressors. This builds a baseline case profile before proceeding to formal risk scales.",
                "domain": "health",
                "verified": True,
                "year": 2024,
                "concept": "clinical_consultation"
            },
            {
                "title": "Columbia-Suicide Severity Rating Scale (C-SSRS) Risk Assessment",
                "content": "GUIDELINE SECTION 3: RISK ASSESSMENT. Suicide risk assessment. Perform a rigorous risk assessment to identify high-risk patients. Evaluate active suicide ideation, intent, plan, and self-harm patterns. Proper risk stratification separates patients into mild, moderate, or severe risk and dictates the timeline for diagnostic interviews.",
                "domain": "health",
                "verified": True,
                "year": 2025,
                "concept": "risk_assessment"
            },
            {
                "title": "DSM-5 Structural Diagnostic Interview Protocols",
                "content": "GUIDELINE SECTION 4: DIAGNOSTIC INTERVIEW. Structured diagnostic interview. Apply formal DSM-5 diagnostic criteria for Major Depressive Disorder (MDD) or PTSD. This structured interview requires checking the presence of 5 or more symptoms (including depressed mood or anhedonia) over a 2-week period, establishing a clear diagnosis before starting therapeutic interventions.",
                "domain": "health",
                "verified": True,
                "year": 2024,
                "concept": "diagnostic_interview"
            },
            {
                "title": "Cognitive Behavioral Therapy (CBT) and Trauma-Informed Interventions",
                "content": "GUIDELINE SECTION 5: GUIDED INTERVENTION. Treatment and CBT. Deliver evidence-based guided interventions, such as Trauma-Informed Cognitive Behavioral Therapy (CBT), sleep hygiene education, or mindfulness-based stress reduction. Ensure this treatment planning is personalized to the patient's specific history of Adverse Childhood Experiences (ACE) and trauma.",
                "domain": "health",
                "verified": True,
                "year": 2025,
                "concept": "trauma_informed_treatment"
            },
            
            # ── DevOps guidelines ──
            {
                "title": "DevOps Guide: CI Build Pipeline Standards",
                "content": "DEVOPS PIPELINE SECTION 1: CI BUILD. Compile source code, run basic linting, and generate build artifacts on every push. Developers must fix any compile errors immediately before build stages are marked green.",
                "domain": "tech",
                "verified": True,
                "year": 2025,
                "concept": "ci_build"
            },
            {
                "title": "QA Practice: Automated Test Suite Implementation",
                "content": "DEVOPS PIPELINE SECTION 2: AUTOMATED TESTING. Execute unit and integration tests automatically on code updates. Test coverage must meet a minimum threshold of 80% with no regressions before the build is promoted to security scanning.",
                "domain": "tech",
                "verified": True,
                "year": 2024,
                "concept": "automated_testing"
            },
            {
                "title": "Security Standard: Static & Dynamic Vulnerability Scanning",
                "content": "DEVOPS PIPELINE SECTION 3: VULNERABILITY SCAN. Scan source code and container images for security vulnerabilities. High or critical severity vulnerabilities must be remediated or officially signed off before staging deployment.",
                "domain": "tech",
                "verified": True,
                "year": 2025,
                "concept": "vulnerability_scan"
            },
            {
                "title": "Release Practice: Staging Validation & Verification Protocols",
                "content": "DEVOPS PIPELINE SECTION 4: STAGING ENVIRONMENT. Deploy build artifacts to a replica staging environment. Execute load tests, user acceptance verification, and environmental health checks before marking release ready.",
                "domain": "tech",
                "verified": True,
                "year": 2024,
                "concept": "staging_environment"
            },
            {
                "title": "Deployment standard: Canary Releases and Production Rollout",
                "content": "DEVOPS PIPELINE SECTION 5: CANARY RELEASE. Deploy release to 5% of production traffic. Monitor error rates, system resource usage, and active logs. Slowly scale traffic to 100% over a 4-hour window if metrics remain stable.",
                "domain": "tech",
                "verified": True,
                "year": 2025,
                "concept": "canary_release"
            }
        ]

        print("Generating embeddings for all guideline documents...")
        for doc_info in docs_data:
            doc_embedding = model.encode([doc_info["content"]])[0].tolist()
            doc = Document(
                title=doc_info["title"],
                content=doc_info["content"],
                domain=doc_info["domain"],
                verified=doc_info["verified"],
                year=doc_info["year"],
                embedding=doc_embedding
            )
            db.add(doc)
            db.flush()

            # Map this document to its corresponding clinical/DevOps concept
            mapping = DocumentKGMapping(
                document_id=doc.id,
                node_id=inserted_nodes[doc_info["concept"]].id
            )
            db.add(mapping)

        db.commit()
        print("Successfully seeded clinical and DevOps guidelines and mapped them to the KG!")

    finally:
        db.close()

if __name__ == "__main__":
    seed_data()

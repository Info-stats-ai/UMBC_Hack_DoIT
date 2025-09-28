from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import os
from typing import Dict, Any, List
import logging
from study_groups_service import StudyGroupsService
import google.generativeai as genai
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UMBC Academic Risk Predictor API",
    description="Predict student success in courses using graph analytics and ML",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = os.path.join("..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    
    # Serve CSS and JS files directly from root
    @app.get("/styles.css")
    async def get_styles():
        return FileResponse(os.path.join(frontend_path, "styles.css"))
    
    @app.get("/script.js")
    async def get_script():
        return FileResponse(os.path.join(frontend_path, "script.js"))
    
    @app.get("/dashboard.css")
    async def get_dashboard_styles():
        return FileResponse(os.path.join(frontend_path, "dashboard.css"))
    
    @app.get("/dashboard.js")
    async def get_dashboard_script():
        return FileResponse(os.path.join(frontend_path, "dashboard.js"))

# Neo4j connection
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Harsh@0603"  # Default password - change this to your Neo4j password
NEO4J_DB = "neo4j"

# Gemini API configuration
GEMINI_API_KEY = "AIzaSyAjU7Bf4p4GpChsdMgpJfIVSM74TdYrL1U"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.0-flash')
logger.info("Gemini API configured successfully")

# Load model (optional for chatbot functionality)
model = None
feature_names = []
try:
    MODEL_PATH = os.path.join("..", "ml", "models", "academic_risk_model_optimized.joblib")
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        feature_names = model_data['feature_names']
        logger.info(f"Loaded model with {len(feature_names)} features")
    else:
        # Fallback to basic model if optimized doesn't exist
        FALLBACK_MODEL_PATH = os.path.join("..", "ml", "models", "academic_risk_model.joblib")
        if os.path.exists(FALLBACK_MODEL_PATH):
            model_data = joblib.load(FALLBACK_MODEL_PATH)
            model = model_data['model']
            feature_names = model_data['feature_names']
            logger.info(f"Loaded fallback model with {len(feature_names)} features")
        else:
            logger.warning("No ML model found. Chatbot will work without ML predictions.")
except Exception as e:
    logger.warning(f"Could not load ML model: {e}. Chatbot will work without ML predictions.")
    model = None
    feature_names = []

# Neo4j driver
driver = None
try:
    # Check if Neo4j is available (optional dependency)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('127.0.0.1', 7687))
    sock.close()
    
    if result == 0:  # Port is open
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        logger.info("Connected to Neo4j database")
        
        # Check dataset loading
        try:
            with driver.session(database=NEO4J_DB) as session:
                total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
                student_count = session.run("MATCH (s:Student) RETURN count(s) as count").single()["count"]
                course_count = session.run("MATCH (c:Course) RETURN count(c) as count").single()["count"]
                
                print(f"DATASET LOADED SUCCESSFULLY!")
                print(f"   Total Nodes: {total_nodes}")
                print(f"   Students: {student_count}")
                print(f"   Courses: {course_count}")
                logger.info(f"Dataset loaded - Total nodes: {total_nodes}, Students: {student_count}, Courses: {course_count}")
                
        except Exception as e:
            print(f"ERROR CHECKING DATASET: {e}")
            logger.error(f"Error checking dataset: {e}")
    else:
        print("NEO4J NOT AVAILABLE: Neo4j server is not running on localhost:7687")
        logger.warning("Neo4j server not available, running without graph database features")
        
except Exception as e:
    driver = None
    logger.error(f"Failed to connect to Neo4j: {e}")
    print(f"NEO4J CONNECTION FAILED: {e}")
    print("CONTINUING WITHOUT NEO4J: Server will run with limited functionality")

# Initialize Study Groups Service
study_groups_service = StudyGroupsService(neo4j_driver=driver)

# Pydantic models
class PredictionRequest(BaseModel):
    student_id: str
    course_id: str

class PredictionResponse(BaseModel):
    student_id: str
    course_id: str
    prediction_result: int  # 0 = high risk, 1 = low risk
    confidence: float
    risk_level: str
    probability: Dict[str, float]
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    neo4j_connected: bool
    features_count: int

class StudyPartnerRequest(BaseModel):
    student_id: str
    course_id: str
    max_partners: int = 10

class StudyPartnerResponse(BaseModel):
    student_id: str
    name: str
    learning_style: str
    current_courses: List[str]
    completed_courses: List[str]
    performance_level: str
    preferred_pace: str
    work_hours: int
    instruction_mode: str
    compatibility_score: float
    compatibility_factors: Dict[str, float]

class StudyGroupResponse(BaseModel):
    group_id: str
    course_id: str
    course_name: str
    members: List[StudyPartnerResponse]
    avg_compatibility: float
    recommended_meeting_time: str
    group_size: int
    learning_style_diversity: float
    performance_balance: float

class StudyGroupsRequest(BaseModel):
    course_id: str
    min_group_size: int = 3
    max_group_size: int = 5

# Helper functions
def get_student_course_features(student_id: str, course_id: str) -> pd.DataFrame:
    """Extract features for a student-course pair from CSV data"""
    # Load features directly from ML data CSV since Neo4j doesn't have embeddings
    try:
        csv_path = os.path.join("..", "ml", "data", "ml_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Filter for the specific student-course pair
            filtered_df = df[(df['student_id'] == student_id) & (df['course_id'] == course_id)]

            if filtered_df.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for student {student_id} and course {course_id}"
                )

            return filtered_df.iloc[:1]  # Return first row as DataFrame
        else:
            # Fallback to Neo4j if CSV doesn't exist (original implementation)
            if not driver:
                raise HTTPException(status_code=503, detail="Neo4j database not available")

            query = """
            MATCH (s:Student {id: $student_id})-[:COMPLETED]->(c:Course {id: $course_id})
            RETURN s.id AS student_id,
                   c.id AS course_id,
                   s.fastRP_embedding AS s_emb,
                   c.fastRP_embedding AS c_emb,
                   s.louvain_community AS s_comm,
                   c.louvain_community AS c_comm
            """

            with driver.session(database=NEO4J_DB) as session:
                result = session.run(query, {"student_id": student_id, "course_id": course_id})
                rows = result.data()

                if not rows:
                    # Try to get features even if no direct COMPLETED relationship
                    fallback_query = """
                    MATCH (s:Student {id: $student_id}), (c:Course {id: $course_id})
                    WHERE s.fastRP_embedding IS NOT NULL AND c.fastRP_embedding IS NOT NULL
                    RETURN s.id AS student_id,
                           c.id AS course_id,
                           s.fastRP_embedding AS s_emb,
                           c.fastRP_embedding AS c_emb,
                           s.louvain_community AS s_comm,
                           c.louvain_community AS c_comm
                    """
                    result = session.run(fallback_query, {"student_id": student_id, "course_id": course_id})
                    rows = result.data()

                if not rows:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No data found for student {student_id} and course {course_id}"
                    )

            df = pd.DataFrame(rows)

            # Expand embeddings into columns
            if 's_emb' in df.columns and df['s_emb'].iloc[0] is not None:
                s_emb_df = pd.DataFrame(df['s_emb'].tolist()).add_prefix('s_emb_')
                df = pd.concat([df, s_emb_df], axis=1)

            if 'c_emb' in df.columns and df['c_emb'].iloc[0] is not None:
                c_emb_df = pd.DataFrame(df['c_emb'].tolist()).add_prefix('c_emb_')
                df = pd.concat([df, c_emb_df], axis=1)

            return df

    except Exception as e:
        if "No data found" in str(e):
            raise
        logger.error(f"Error loading features: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load features: {str(e)}")

def calculate_instruction_mode_compatibility(course_modes, student_preferred_mode: str) -> int:
    """Calculate compatibility between course instruction modes and student preference"""
    # If no course modes or student preference, return None (no score)
    if not course_modes or not student_preferred_mode:
        return None
    
    # Handle both string and list inputs
    if isinstance(course_modes, list):
        available_modes = [str(mode).strip() for mode in course_modes]
    else:
        available_modes = [str(course_modes).strip()]
    
    # Filter out empty or "Not specified" modes
    available_modes = [mode for mode in available_modes if mode and mode.lower() not in ['not specified', 'none', '']]
    
    # If no valid modes after filtering, return None
    if not available_modes:
        return None
    
    # Direct match gets highest score
    if student_preferred_mode in available_modes:
        return 100
    
    # Check for similar modes
    mode_similarity = {
        'Online': ['Hybrid'],
        'Hybrid': ['Online', 'In-person'],
        'In-person': ['Hybrid']
    }
    
    similar_modes = mode_similarity.get(student_preferred_mode, [])
    for mode in available_modes:
        if mode in similar_modes:
            return 80
    
    return 60  # Lower score for incompatible modes

def calculate_faculty_compatibility(faculty_teaching_style, student_learning_style: str) -> int:
    """Calculate compatibility between faculty teaching style and student learning style"""
    if not faculty_teaching_style:
        return 60  # Neutral score for unknown teaching style
    
    # Handle both string and list inputs
    if isinstance(faculty_teaching_style, list):
        faculty_styles = [str(style).strip() for style in faculty_teaching_style]
    else:
        faculty_styles = [style.strip() for style in str(faculty_teaching_style).split(',')]
    
    # Direct matches
    if student_learning_style in faculty_teaching_style:
        return 95
    
    # Specific style matches
    style_matches = {
        'Visual': ['Visual', 'Lecture', 'Presentation', 'Demonstration'],
        'Auditory': ['Auditory', 'Lecture', 'Discussion', 'Socratic'],
        'Kinesthetic': ['Kinesthetic', 'Hands-on', 'Project-Based', 'Lab', 'Workshop'],
        'Reading-Writing': ['Reading', 'Writing', 'Text-based', 'Written']
    }
    
    student_styles = style_matches.get(student_learning_style, [])
    
    # Check for partial matches
    for faculty_style in faculty_styles:
        for student_style in student_styles:
            if student_style.lower() in faculty_style.lower():
                return 85
    
    # Check for complementary styles
    complementary_styles = {
        'Visual': ['Lecture', 'Presentation', 'Demonstration'],
        'Auditory': ['Lecture', 'Discussion', 'Socratic'],
        'Kinesthetic': ['Project-Based', 'Lab', 'Workshop', 'Hands-on'],
        'Reading-Writing': ['Text-based', 'Written', 'Reading']
    }
    
    student_complementary = complementary_styles.get(student_learning_style, [])
    for faculty_style in faculty_styles:
        for comp_style in student_complementary:
            if comp_style.lower() in faculty_style.lower():
                return 75
    
    return 40  # Low compatibility for no matches

def generate_recommendations(prediction_result: int, confidence: float) -> List[str]:
    """Generate recommendations based on prediction"""
    if prediction_result == 1:  # Low risk (success predicted)
        recommendations = [
            "Student shows strong potential for success in this course",
            "Consider advanced coursework or additional challenges",
            "Monitor progress and provide enrichment opportunities"
        ]
    else:  # High risk
        recommendations = [
            "Schedule early intervention meeting with academic advisor",
            "Recommend prerequisite review or foundational course completion",
            "Consider tutoring or study group participation",
            "Monitor attendance and engagement closely",
            "Provide additional resources and support materials"
        ]
    
    if confidence < 0.7:
        recommendations.append("Low confidence prediction - gather more student data")
    
    return recommendations

# API endpoints
@app.get("/")
async def root():
    """Serve the frontend homepage"""
    frontend_file = os.path.join("..", "frontend", "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {"message": "UMBC Academic Risk Predictor API", "status": "running", "frontend": "not found"}

@app.get("/student-options")
async def serve_student_options():
    """Serve the student options page"""
    options_file = os.path.join("..", "frontend", "student-options.html")
    if os.path.exists(options_file):
        return FileResponse(options_file)
    return {"message": "Student options page not found"}

@app.get("/student-options.html")
async def serve_student_options_html():
    """Serve the student options page with .html extension"""
    options_file = os.path.join("..", "frontend", "student-options.html")
    if os.path.exists(options_file):
        return FileResponse(options_file)
    return {"message": "Student options page not found"}

@app.get("/student-options.js")
async def get_student_options_script():
    return FileResponse(os.path.join("..", "frontend", "student-options.js"))

@app.get("/study-groups")
async def serve_study_groups():
    """Serve the study groups page"""
    study_groups_file = os.path.join("..", "frontend", "study-groups.html")
    if os.path.exists(study_groups_file):
        return FileResponse(study_groups_file)
    return {"message": "Study groups page not found"}

@app.get("/study-groups.html")
async def serve_study_groups_html():
    """Serve the study groups page with .html extension"""
    study_groups_file = os.path.join("..", "frontend", "study-groups.html")
    if os.path.exists(study_groups_file):
        return FileResponse(study_groups_file)
    return {"message": "Study groups page not found"}

@app.get("/study-groups.css")
async def get_study_groups_styles():
    return FileResponse(os.path.join("..", "frontend", "study-groups.css"))

@app.get("/study-groups.js")
async def get_study_groups_script():
    return FileResponse(os.path.join("..", "frontend", "study-groups.js"))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model and driver else "degraded",
        model_loaded=model is not None,
        neo4j_connected=driver is not None,
        features_count=len(feature_names) if feature_names else 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_student_risk(request: PredictionRequest):
    """Predict academic risk for a student-course pair"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get features from Neo4j
        df = get_student_course_features(request.student_id, request.course_id)
        
        # Prepare features for prediction
        feature_data = []
        for feature in feature_names:
            if feature in df.columns:
                feature_data.append(df[feature].iloc[0])
            else:
                feature_data.append(0.0)  # Default value for missing features
        
        # Make prediction
        X = np.array([feature_data])
        prediction_proba = model.predict_proba(X)[0]
        prediction_result = model.predict(X)[0]
        confidence = max(prediction_proba)
        
        # Determine risk level
        risk_level = "Low Risk (High Success Expected)" if prediction_result == 1 else "High Risk (Support Needed)"
        
        # Generate recommendations
        recommendations = generate_recommendations(prediction_result, confidence)
        
        return PredictionResponse(
            student_id=request.student_id,
            course_id=request.course_id,
            prediction_result=int(prediction_result),
            confidence=float(confidence),
            risk_level=risk_level,
            probability={
                "high_risk": float(prediction_proba[0]),
                "low_risk": float(prediction_proba[1])
            },
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/students")
async def get_available_students():
    """Get list of available students"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    query = "MATCH (s:Student) RETURN s.id as student_id ORDER BY s.id LIMIT 20"
    
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(query)
        students = [record["student_id"] for record in result]
    
    return {"students": students}

@app.get("/courses")
async def get_available_courses():
    """Get list of available courses"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    query = "MATCH (c:Course) RETURN c.id as course_id ORDER BY c.id LIMIT 20"
    
    with driver.session(database=NEO4J_DB) as session:
        result = session.run(query)
        courses = [record["course_id"] for record in result]
    
    return {"courses": courses}

# Study Groups API endpoints
@app.post("/study-partners", response_model=List[StudyPartnerResponse])
async def find_study_partners(request: StudyPartnerRequest):
    """Find compatible study partners for a student in a course"""
    try:
        partners = study_groups_service.find_study_partners(
            request.student_id, 
            request.course_id, 
            request.max_partners
        )
        
        # Convert to response format
        partner_responses = []
        for partner in partners:
            partner_response = StudyPartnerResponse(
                student_id=partner.student_id,
                name=partner.name,
                learning_style=partner.learning_style,
                current_courses=partner.current_courses,
                completed_courses=partner.completed_courses,
                performance_level=partner.performance_level,
                preferred_pace=partner.preferred_pace,
                work_hours=partner.work_hours,
                instruction_mode=partner.instruction_mode,
                compatibility_score=partner.compatibility_score,
                compatibility_factors=partner.compatibility_factors
            )
            partner_responses.append(partner_response)
        
        return partner_responses
        
    except Exception as e:
        logger.error(f"Error finding study partners: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find study partners: {str(e)}")

@app.post("/study-groups", response_model=List[StudyGroupResponse])
async def create_study_groups(request: StudyGroupsRequest):
    """Create optimal study groups for a course"""
    try:
        groups = study_groups_service.create_study_groups(
            request.course_id,
            request.min_group_size,
            request.max_group_size
        )
        
        # Convert to response format
        group_responses = []
        for group in groups:
            member_responses = []
            for member in group.members:
                member_response = StudyPartnerResponse(
                    student_id=member.student_id,
                    name=member.name,
                    learning_style=member.learning_style,
                    current_courses=member.current_courses,
                    completed_courses=member.completed_courses,
                    performance_level=member.performance_level,
                    preferred_pace=member.preferred_pace,
                    work_hours=member.work_hours,
                    instruction_mode=member.instruction_mode,
                    compatibility_score=member.compatibility_score,
                    compatibility_factors=member.compatibility_factors
                )
                member_responses.append(member_response)
            
            group_response = StudyGroupResponse(
                group_id=group.group_id,
                course_id=group.course_id,
                course_name=group.course_name,
                members=member_responses,
                avg_compatibility=group.avg_compatibility,
                recommended_meeting_time=group.recommended_meeting_time,
                group_size=group.group_size,
                learning_style_diversity=group.learning_style_diversity,
                performance_balance=group.performance_balance
            )
            group_responses.append(group_response)
        
        return group_responses
        
    except Exception as e:
        logger.error(f"Error creating study groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create study groups: {str(e)}")

@app.get("/study-groups/course/{course_id}")
async def get_study_groups_for_course(course_id: str):
    """Get existing study groups for a specific course"""
    try:
        groups = study_groups_service.create_study_groups(course_id)
        
        return {
            "course_id": course_id,
            "total_groups": len(groups),
            "groups": [
                {
                    "group_id": group.group_id,
                    "member_count": group.group_size,
                    "avg_compatibility": round(group.avg_compatibility, 2),
                    "meeting_time": group.recommended_meeting_time,
                    "learning_diversity": round(group.learning_style_diversity, 2)
                }
                for group in groups
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting study groups for course {course_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get study groups: {str(e)}")

# Dashboard API endpoints
@app.get("/dashboard")
async def serve_dashboard():
    """Serve the advisor dashboard"""
    dashboard_file = os.path.join("..", "frontend", "dashboard.html")
    if os.path.exists(dashboard_file):
        return FileResponse(dashboard_file)
    return {"message": "Dashboard not found"}

@app.get("/dashboard/overview")
async def get_dashboard_overview():
    """Get dashboard overview data"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    # Get basic statistics
    with driver.session(database=NEO4J_DB) as session:
        # Total students
        student_count = session.run("MATCH (s:Student) RETURN count(s) as count").single()["count"]
        
        # Courses count
        course_count = session.run("MATCH (c:Course) RETURN count(c) as count").single()["count"]
        
        # Completed courses (for success rate calculation)
        completed_count = session.run("MATCH ()-[r:COMPLETED]->() RETURN count(r) as count").single()["count"]
    
    return {
        "total_students": student_count,
        "total_courses": course_count,
        "completed_courses": completed_count,
        "success_rate": 87.0,  # This would be calculated from actual grade data
        "avg_gpa": 3.2
    }

@app.get("/dashboard/at-risk-students")
async def get_at_risk_students():
    """Get at-risk students analysis"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    # This would use the trained model to identify at-risk students
    # For now, return sample data
    at_risk_students = [
        {
            "student_id": "ZO28124",
            "name": "Alex Johnson",
            "major": "Computer Science",
            "risk_level": "high",
            "current_gpa": 2.8,
            "risk_factors": ["Low GPA", "Multiple Failed Courses"],
            "recommendations": ["Immediate Advisor Meeting", "Academic Support Services"]
        },
        {
            "student_id": "XE28807",
            "name": "David Wilson",
            "major": "Engineering",
            "risk_level": "high",
            "current_gpa": 2.7,
            "risk_factors": ["Academic Probation", "Missing Prerequisites"],
            "recommendations": ["Tutoring", "Prerequisite Review"]
        },
        {
            "student_id": "EY56522",
            "name": "Michael Brown",
            "major": "Biology",
            "risk_level": "medium",
            "current_gpa": 2.9,
            "risk_factors": ["Declining Performance"],
            "recommendations": ["Monitor Closely", "Study Skills Workshop"]
        }
    ]
    
    return {
        "high_risk": len([s for s in at_risk_students if s["risk_level"] == "high"]),
        "medium_risk": len([s for s in at_risk_students if s["risk_level"] == "medium"]),
        "low_risk": len([s for s in at_risk_students if s["risk_level"] == "low"]),
        "students": at_risk_students
    }

@app.get("/dashboard/course-paths/{major}")
async def get_course_paths(major: str):
    """Get optimal course paths for a major"""
    
    # Sample course paths - in reality, this would use graph algorithms
    paths = {
        "computer-science": {
            "total_credits": 120,
            "estimated_duration": "4 years",
            "success_probability": 87,
            "path": [
                {"semester": "Fall Year 1", "courses": ["CSEE 200", "MATH 151", "ENGL 100"]},
                {"semester": "Spring Year 1", "courses": ["CSEE 201", "MATH 152", "PHYS 121"]},
                {"semester": "Fall Year 2", "courses": ["CSEE 301", "MATH 251", "CSEE 300"]},
                {"semester": "Spring Year 2", "courses": ["CSEE 302", "CSEE 400", "MATH 301"]}
            ],
            "alternatives": [
                {
                    "name": "Accelerated Path",
                    "description": "Complete degree in 3 years with summer courses",
                    "pros": ["Faster completion"],
                    "cons": ["Higher workload"],
                    "duration": "3 years"
                },
                {
                    "name": "Research Focus",
                    "description": "Include undergraduate research opportunities",
                    "pros": ["Research experience", "Graduate school prep"],
                    "cons": ["Longer timeline"],
                    "duration": "4.5 years"
                }
            ]
        },
        "mathematics": {
            "total_credits": 120,
            "estimated_duration": "4 years",
            "success_probability": 91,
            "path": [
                {"semester": "Fall Year 1", "courses": ["MATH 151", "PHYS 121", "ENGL 100"]},
                {"semester": "Spring Year 1", "courses": ["MATH 152", "MATH 221", "CSEE 200"]},
                {"semester": "Fall Year 2", "courses": ["MATH 251", "MATH 301", "MATH 221"]},
                {"semester": "Spring Year 2", "courses": ["MATH 302", "MATH 401", "STAT 355"]}
            ],
            "alternatives": []
        }
    }
    
    return paths.get(major, paths["computer-science"])

@app.get("/dashboard/study-groups")
async def get_study_groups():
    """Get study group recommendations"""
    
    # Sample study groups - in reality, this would use clustering algorithms
    study_groups = [
        {
            "name": "Group Alpha",
            "course": "CSEE 200",
            "members": [
                {"id": "STU001", "name": "Student A", "learning_style": "Visual"},
                {"id": "STU002", "name": "Student B", "learning_style": "Auditory"},
                {"id": "STU003", "name": "Student C", "learning_style": "Kinesthetic"}
            ],
            "compatibility": 92,
            "avg_gpa": 3.2,
            "meeting_time": "Tue/Thu 2-4 PM",
            "location": "Library Study Room 1"
        },
        {
            "name": "Group Beta",
            "course": "MATH 151",
            "members": [
                {"id": "STU004", "name": "Student D", "learning_style": "Visual"},
                {"id": "STU005", "name": "Student E", "learning_style": "Reading"},
                {"id": "STU006", "name": "Student F", "learning_style": "Kinesthetic"}
            ],
            "compatibility": 88,
            "avg_gpa": 3.4,
            "meeting_time": "Mon/Wed 3-5 PM",
            "location": "Math Lab"
        }
    ]
    
    return {
        "groups": study_groups,
        "learning_style_distribution": {
            "visual": 35,
            "auditory": 25,
            "kinesthetic": 30,
            "reading": 10
        }
    }

@app.get("/dashboard/predictions")
async def get_predictions():
    """Get AI predictions and insights"""
    
    # Sample predictions - in reality, this would use the trained model
    predictions = {
        "early_warning_signals": {
            "count": 15,
            "description": "Students showing early signs of academic struggle",
            "recommended_action": "Immediate intervention"
        },
        "success_predictors": {
            "prerequisite_completion": {
                "success_rate": 89,
                "description": "Students with strong prerequisite completion show high success rate"
            }
        },
        "course_sequencing": {
            "improvement_potential": 12,
            "description": "Alternative course sequences could improve success rates"
        },
        "success_probability_by_course": [
            {"course": "CSEE 200", "probability": 89},
            {"course": "MATH 151", "probability": 76},
            {"course": "BIOL 141", "probability": 82},
            {"course": "CSEE 201", "probability": 85},
            {"course": "MATH 152", "probability": 71}
        ],
        "risk_factors": [
            {"factor": "Low GPA", "percentage": 35},
            {"factor": "Missing Prerequisites", "percentage": 25},
            {"factor": "Course Load", "percentage": 20},
            {"factor": "Attendance", "percentage": 15},
            {"factor": "Other", "percentage": 5}
        ]
    }
    
    return predictions

# Course Planning API endpoints
@app.get("/api/student-info/{student_id}")
async def get_student_info(student_id: str):
    """Get comprehensive student information for course planning"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")

    try:
        with driver.session(database=NEO4J_DB) as session:
            # Get student basic info and current enrollment
            student_query = """
            MATCH (s:Student {id: $student_id})
            OPTIONAL MATCH (s)-[:PURSUING]->(d:Degree)
            OPTIONAL MATCH (s)-[:COMPLETED]->(completed_courses:Course)
            RETURN s.id as student_id,
                   s.learningStyle as learning_style,
                   s.preferredCourseLoad as preferred_course_load,
                   s.preferredInstructionMode as instruction_mode,
                   s.preferredPace as preferred_pace,
                   s.enrollmentDate as enrollment_date,
                   s.expectedGraduation as expected_graduation,
                   d.name as degree,
                   count(completed_courses) as completed_courses_count
            """

            result = session.run(student_query, {"student_id": student_id})
            student_record = result.single()

            if not student_record:
                raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

            return {
                "student_id": student_record["student_id"],
                "learning_style": student_record["learning_style"],
                "preferred_course_load": student_record["preferred_course_load"],
                "instruction_mode": student_record["instruction_mode"],
                "preferred_pace": student_record["preferred_pace"],
                "enrollment_date": student_record["enrollment_date"],
                "expected_graduation": student_record["expected_graduation"],
                "degree": student_record["degree"],
                "completed_courses_count": student_record["completed_courses_count"]
            }

    except Exception as e:
        logger.error(f"Error getting student info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/terms")
async def get_available_terms():
    """Get all available terms for course planning"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")

    try:
        with driver.session(database=NEO4J_DB) as session:
            terms_query = """
            MATCH (t:Term)
            RETURN t.id as id, t.name as name, t.type as type
            ORDER BY t.name
            """

            result = session.run(terms_query)
            terms = []
            for record in result:
                terms.append({
                    "id": record["id"],
                    "name": record["name"],
                    "status": "available"
                })

            return terms

    except Exception as e:
        logger.error(f"Error getting terms: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class CourseRecommendationRequest(BaseModel):
    student_id: str
    term_id: str

class ChatbotRequest(BaseModel):
    student_id: str
    question: str

class ChatbotResponse(BaseModel):
    student_id: str
    question: str
    answer: str
    recommendations: List[Dict[str, Any]]
    confidence: float

@app.post("/api/course-recommendations")
async def get_course_recommendations(request: CourseRecommendationRequest):
    """Get course recommendations for a student in a specific term"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")

    try:
        with driver.session(database=NEO4J_DB) as session:
            # Get student's completed courses and preferences
            student_data_query = """
            MATCH (s:Student {id: $student_id})
            OPTIONAL MATCH (s)-[:COMPLETED]->(completed:Course)
            OPTIONAL MATCH (s)-[:PURSUING]->(d:Degree)
            RETURN s.learningStyle as learning_style,
                   s.preferredCourseLoad as preferred_course_load,
                   collect(completed.id) as completed_courses,
                   d.name as degree
            """

            student_result = session.run(student_data_query, {"student_id": request.student_id})
            student_data = student_result.single()

            if not student_data:
                raise HTTPException(status_code=404, detail=f"Student {request.student_id} not found")

            completed_courses = student_data["completed_courses"] or []
            learning_style = student_data["learning_style"]

            # Get courses offered in the selected term, prioritizing degree-relevant courses
            courses_query = """
            MATCH (c:Course)-[:OFFERED_IN]->(t:Term {id: $term_id})
            WHERE NOT c.id IN $completed_courses
            OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(future_course:Course)
            OPTIONAL MATCH (prereq:Course)-[:PREREQUISITE_FOR]->(c)
            OPTIONAL MATCH (f:Faculty)-[:TEACHES]->(c)
            OPTIONAL MATCH (c)-[:REQUIRED_FOR]->(d:Degree)
            WHERE d.name = $degree_name OR d.name IS NULL
            RETURN c.id as course_id,
                   c.name as course_name,
                   c.credits as credits,
                   c.difficulty as difficulty,
                   c.instructionMode as instruction_mode,
                   collect(DISTINCT future_course.id) as leads_to,
                   collect(DISTINCT future_course.name) as leads_to_names,
                   collect(DISTINCT prereq.id) as prerequisites,
                   collect(DISTINCT prereq.name) as prerequisite_names,
                   collect(DISTINCT {
                       name: f.name,
                       teachingStyle: f.teachingStyle
                   }) as faculty_options,
                   CASE 
                       WHEN c.id STARTS WITH 'CS' AND $degree_name CONTAINS 'Computer Science' THEN 100
                       WHEN c.id STARTS WITH 'MATH' AND $degree_name CONTAINS 'Computer Science' THEN 90
                       WHEN c.id STARTS WITH 'ENGL' AND $degree_name CONTAINS 'Computer Science' THEN 80
                       WHEN c.id STARTS WITH 'PHYS' AND $degree_name CONTAINS 'Computer Science' THEN 70
                       ELSE 50
                   END as degree_relevance_score
            ORDER BY degree_relevance_score DESC, c.id
            """

            courses_result = session.run(courses_query, {
                "term_id": request.term_id,
                "completed_courses": completed_courses,
                "learning_style": learning_style,
                "degree_name": student_data["degree"] or ""
            })

            recommendations = []
            for record in courses_result:
                course_id = record["course_id"]
                prerequisites = record["prerequisites"] or []
                prerequisite_names = record["prerequisite_names"] or []
                leads_to = record["leads_to"] or []
                leads_to_names = record["leads_to_names"] or []
                faculty_options = record["faculty_options"] or []
                instruction_mode = record["instruction_mode"]

                # Check if prerequisites are met
                missing_prerequisites = [p for p in prerequisites if p not in completed_courses]
                is_blocked = len(missing_prerequisites) > 0

                # Calculate priority score
                degree_relevance = record["degree_relevance_score"] or 50
                priority_score = degree_relevance  # Start with degree relevance

                # Higher priority for prerequisite courses
                if len(leads_to) > 0:
                    priority_score += 20

                # Lower priority if prerequisites missing
                if is_blocked:
                    priority_score -= 30

                # Calculate faculty compatibility for each faculty member
                enhanced_faculty_options = []
                best_faculty_compatibility = 50
                
                if faculty_options:
                    for faculty in faculty_options:
                        compatibility = calculate_faculty_compatibility(
                            faculty.get("teachingStyle", ""), 
                            learning_style
                        )
                        enhanced_faculty_options.append({
                            "name": faculty.get("name", "TBA"),
                            "teachingStyle": faculty.get("teachingStyle", "Not specified"),
                            "compatibility": compatibility
                        })
                        best_faculty_compatibility = max(best_faculty_compatibility, compatibility)

                # Calculate instruction mode compatibility
                instruction_mode_compatibility = calculate_instruction_mode_compatibility(
                    instruction_mode, 
                    student_data.get("preferred_instruction_mode", "")
                )

                # Recommendation reason
                reason = "Available for registration"
                if degree_relevance >= 90:
                    reason = "Core requirement for your degree"
                elif degree_relevance >= 80:
                    reason = "Important for your degree program"
                elif len(leads_to) > 0:
                    reason = f"Prerequisite for {len(leads_to)} advanced courses"
                if is_blocked:
                    reason = f"Missing prerequisites: {', '.join(missing_prerequisites)}"

                # Format instruction modes for display
                if isinstance(instruction_mode, list):
                    formatted_instruction_modes = ", ".join(instruction_mode)
                else:
                    formatted_instruction_modes = str(instruction_mode) if instruction_mode else "Not specified"

                recommendations.append({
                    "course_id": course_id,
                    "course_name": record["course_name"],
                    "credits": record["credits"] or 3,
                    "difficulty": record["difficulty"],
                    "instruction_mode": formatted_instruction_modes,
                    "instruction_mode_compatibility": instruction_mode_compatibility,
                    "is_prerequisite": len(leads_to) > 0,
                    "prerequisite_for": leads_to,
                    "prerequisite_for_names": leads_to_names,
                    "missing_prerequisites": missing_prerequisites,
                    "missing_prerequisite_names": [prerequisite_names[i] for i, prereq in enumerate(prerequisites) if prereq in missing_prerequisites],
                    "is_blocked": is_blocked,
                    "faculty_options": enhanced_faculty_options,
                    "best_faculty_compatibility": best_faculty_compatibility,
                    "priority_score": max(0, min(100, priority_score)),
                    "recommendation_reason": reason,
                    "degree_relevance": degree_relevance
                })

            return recommendations

    except Exception as e:
        logger.error(f"Error getting course recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Course Planning Routes
@app.get("/course-planning")
async def serve_course_planning():
    """Serve the course planning page"""
    planning_file = os.path.join("..", "frontend", "course-planning.html")
    if os.path.exists(planning_file):
        return FileResponse(planning_file)
    return {"message": "Course planning page not found"}

@app.get("/course-planning.html")
async def serve_course_planning_html():
    """Serve the course planning page with .html extension"""
    planning_file = os.path.join("..", "frontend", "course-planning.html")
    if os.path.exists(planning_file):
        return FileResponse(planning_file)
    return {"message": "Course planning page not found"}

@app.get("/course-planning.js")
async def get_course_planning_script():
    return FileResponse(os.path.join("..", "frontend", "course-planning.js"))

@app.get("/course-planning.css")
async def get_course_planning_styles():
    return FileResponse(os.path.join("..", "frontend", "course-planning.css"))

# AI Advisory Chatbot API endpoints
@app.get("/ai-advisory")
async def serve_ai_advisory():
    """Serve the AI advisory chatbot page"""
    advisory_file = os.path.join("..", "frontend", "AI_advisory.html")
    if os.path.exists(advisory_file):
        return FileResponse(advisory_file)
    return {"message": "AI advisory page not found"}

@app.get("/ai-advisory.html")
async def serve_ai_advisory_html():
    """Serve the AI advisory page with .html extension"""
    advisory_file = os.path.join("..", "frontend", "AI_advisory.html")
    if os.path.exists(advisory_file):
        return FileResponse(advisory_file)
    return {"message": "AI advisory page not found"}

@app.get("/AI_advisory.js")
async def get_ai_advisory_script():
    return FileResponse(os.path.join("..", "frontend", "AI_advisory.js"))

@app.get("/AI_advisory.css")
async def get_ai_advisory_styles():
    return FileResponse(os.path.join("..", "frontend", "AI_advisory.css"))

def parse_student_question(question: str) -> Dict[str, Any]:
    """Parse student question and extract intent and entities"""
    question_lower = question.lower()
    
    # Course recommendation patterns
    if any(word in question_lower for word in ['course', 'class', 'take', 'enroll', 'register']):
        if any(word in question_lower for word in ['next', 'upcoming', 'fall', 'spring', 'semester']):
            return {"intent": "course_recommendation", "timeframe": "upcoming", "entities": []}
        elif any(word in question_lower for word in ['difficult', 'hard', 'easy', 'challenging']):
            return {"intent": "course_difficulty", "entities": []}
        elif any(word in question_lower for word in ['professor', 'teacher', 'instructor']):
            return {"intent": "faculty_recommendation", "entities": []}
        else:
            return {"intent": "course_general", "entities": []}
    
    # Degree progress patterns
    elif any(word in question_lower for word in ['graduate', 'graduation', 'credits', 'gpa', 'progress']):
        return {"intent": "degree_progress", "entities": []}
    
    # Prerequisite patterns
    elif any(word in question_lower for word in ['prerequisite', 'prereq', 'need', 'required']):
        return {"intent": "prerequisites", "entities": []}
    
    # General academic advice
    else:
        return {"intent": "general_advice", "entities": []}

def get_student_data(student_id: str) -> Dict[str, Any]:
    """Get comprehensive student data from Neo4j"""
    if not driver:
        return {}
    
    try:
        with driver.session(database=NEO4J_DB) as session:
            # Get student's basic info, completed courses, and degree
            student_query = """
            MATCH (s:Student {id: $student_id})
            OPTIONAL MATCH (s)-[:COMPLETED]->(completed:Course)
            OPTIONAL MATCH (s)-[:PURSUING]->(d:Degree)
            OPTIONAL MATCH (s)-[:ENROLLED_IN]->(enrolled:Course)
            RETURN s.id as student_id,
                   s.learningStyle as learning_style,
                   s.preferredCourseLoad as preferred_course_load,
                   s.preferredInstructionMode as instruction_mode,
                   s.preferredPace as preferred_pace,
                   s.enrollmentDate as enrollment_date,
                   s.expectedGraduation as expected_graduation,
                   d.name as degree,
                   collect(DISTINCT completed.id) as completed_courses,
                   collect(DISTINCT enrolled.id) as enrolled_courses,
                   count(completed) as completed_count
            """
            
            result = session.run(student_query, {"student_id": student_id})
            student_data = result.single()
            
            if not student_data:
                return {}
            
            return {
                "student_id": student_data["student_id"],
                "learning_style": student_data["learning_style"],
                "preferred_course_load": student_data["preferred_course_load"],
                "instruction_mode": student_data["instruction_mode"],
                "preferred_pace": student_data["preferred_pace"],
                "enrollment_date": student_data["enrollment_date"],
                "expected_graduation": student_data["expected_graduation"],
                "degree": student_data["degree"],
                "completed_courses": student_data["completed_courses"] or [],
                "enrolled_courses": student_data["enrolled_courses"] or [],
                "completed_count": student_data["completed_count"]
            }
            
    except Exception as e:
        logger.error(f"Error getting student data: {e}")
        return {}

def find_similar_students(student_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Find students with similar academic profiles using Neo4j similarity"""
    if not driver:
        return []
    
    try:
        with driver.session(database=NEO4J_DB) as session:
            # Simplified similarity search - find students with same degree and learning style
            similarity_query = """
            MATCH (s1:Student {id: $student_id})
            MATCH (s2:Student)
            WHERE s1 <> s2
            OPTIONAL MATCH (s1)-[:PURSUING]->(d1:Degree)
            OPTIONAL MATCH (s2)-[:PURSUING]->(d2:Degree)
            OPTIONAL MATCH (s2)-[:COMPLETED]->(c2:Course)
            
            WITH s1, s2, 
                 d1.name as s1_degree,
                 d2.name as s2_degree,
                 s1.learningStyle as s1_style,
                 s2.learningStyle as s2_style,
                 collect(DISTINCT c2.id) as s2_courses
            
            // Calculate basic similarity scores
            WITH s1, s2, s1_degree, s2_degree, s1_style, s2_style, s2_courses,
                 CASE WHEN s1_degree = s2_degree THEN 1.0 ELSE 0.0 END as degree_similarity,
                 CASE WHEN s1_style = s2_style THEN 1.0 ELSE 0.0 END as style_similarity
            
            // Calculate overall similarity score
            WITH s1, s2, 
                 (degree_similarity * 0.6 + style_similarity * 0.4) as overall_similarity,
                 s2_courses, s1_degree, s2_degree, s1_style, s2_style
            
            WHERE overall_similarity > 0.3  // Find students with some similarity
            
            RETURN s2.id as similar_student_id,
                   s2.learningStyle as similar_learning_style,
                   s2.preferredCourseLoad as similar_course_load,
                   s2_courses as similar_completed_courses,
                   overall_similarity,
                   s2_degree as similar_degree
            ORDER BY overall_similarity DESC
            LIMIT $limit
            """
            
            result = session.run(similarity_query, {"student_id": student_id, "limit": limit})
            similar_students = []
            
            for record in result:
                similar_students.append({
                    "student_id": record["similar_student_id"],
                    "learning_style": record["similar_learning_style"],
                    "course_load": record["similar_course_load"],
                    "completed_courses": record["similar_completed_courses"] or [],
                    "similarity_score": record["overall_similarity"],
                    "degree": record["similar_degree"]
                })
            
            return similar_students
            
    except Exception as e:
        logger.error(f"Error finding similar students: {e}")
        return []

def get_courses_from_similar_students(student_id: str, similar_students: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Get course recommendations based on what similar students took"""
    if not driver or not similar_students:
        return []
    
    try:
        student_data = get_student_data(student_id)
        completed_courses = student_data.get("completed_courses", [])
        
        # Get courses that similar students took but current student hasn't
        similar_course_ids = []
        for similar_student in similar_students:
            similar_course_ids.extend(similar_student.get("completed_courses", []))
        
        # Remove duplicates and courses already taken by current student
        similar_course_ids = list(set(similar_course_ids) - set(completed_courses))
        
        if not similar_course_ids:
            return []
        
        with driver.session(database=NEO4J_DB) as session:
            courses_query = """
            MATCH (c:Course)
            WHERE c.id IN $similar_course_ids
            OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(future_course:Course)
            OPTIONAL MATCH (prereq:Course)-[:PREREQUISITE_FOR]->(c)
            OPTIONAL MATCH (f:Faculty)-[:TEACHES]->(c)
            OPTIONAL MATCH (c)-[:REQUIRED_FOR]->(d:Degree)
            WHERE d.name = $degree_name OR d.name IS NULL
            RETURN c.id as course_id,
                   c.name as course_name,
                   c.credits as credits,
                   c.difficulty as difficulty,
                   c.instructionMode as instruction_mode,
                   c.description as description,
                   collect(DISTINCT future_course.id) as leads_to,
                   collect(DISTINCT future_course.name) as leads_to_names,
                   collect(DISTINCT prereq.id) as prerequisites,
                   collect(DISTINCT prereq.name) as prerequisite_names,
                   collect(DISTINCT {
                       name: f.name,
                       teachingStyle: f.teachingStyle
                   }) as faculty_options,
                   CASE 
                       WHEN c.id STARTS WITH 'CSEE' AND $degree_name CONTAINS 'Computer Science' THEN 100
                       WHEN c.id STARTS WITH 'MATH' AND $degree_name CONTAINS 'Computer Science' THEN 90
                       WHEN c.id STARTS WITH 'ENGL' AND $degree_name CONTAINS 'Computer Science' THEN 80
                       WHEN c.id STARTS WITH 'PHYS' AND $degree_name CONTAINS 'Computer Science' THEN 70
                       ELSE 50
                   END as degree_relevance_score
            ORDER BY degree_relevance_score DESC, c.id
            LIMIT $limit
            """
            
            courses_result = session.run(courses_query, {
                "similar_course_ids": similar_course_ids,
                "degree_name": student_data.get("degree", ""),
                "limit": limit
            })
            
            courses = []
            for record in courses_result:
                course_id = record["course_id"]
                prerequisites = record["prerequisites"] or []
                leads_to = record["leads_to"] or []
                faculty_options = record["faculty_options"] or []
                
                # Check if prerequisites are met
                missing_prerequisites = [p for p in prerequisites if p not in completed_courses]
                is_available = len(missing_prerequisites) == 0
                
                courses.append({
                    "course_id": course_id,
                    "course_name": record["course_name"],
                    "credits": record["credits"] or 3,
                    "difficulty": record["difficulty"],
                    "instruction_mode": record["instruction_mode"],
                    "description": record["description"],
                    "is_available": is_available,
                    "missing_prerequisites": missing_prerequisites,
                    "missing_prerequisite_names": [record["prerequisite_names"][i] for i, prereq in enumerate(prerequisites) if prereq in missing_prerequisites],
                    "leads_to": leads_to,
                    "leads_to_names": record["leads_to_names"],
                    "leads_to_count": len(leads_to),
                    "faculty_options": faculty_options,
                    "faculty_count": len(faculty_options),
                    "relevance_score": record["degree_relevance_score"],
                    "recommendation_source": "similar_students"
                })
            
            return courses
            
    except Exception as e:
        logger.error(f"Error getting courses from similar students: {e}")
        return []

def get_available_courses(student_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get available courses for a student with similarity-based recommendations"""
    if not driver:
        return []
    
    try:
        # First, find similar students
        similar_students = find_similar_students(student_id, limit=3)
        
        # Get courses from similar students
        similar_courses = get_courses_from_similar_students(student_id, similar_students, limit=3)
        
        # Get regular course recommendations
        student_data = get_student_data(student_id)
        completed_courses = student_data.get("completed_courses", [])
        
        with driver.session(database=NEO4J_DB) as session:
            # Get all courses with their details
            courses_query = """
            MATCH (c:Course)
            WHERE NOT c.id IN $completed_courses
            OPTIONAL MATCH (c)-[:PREREQUISITE_FOR]->(future_course:Course)
            OPTIONAL MATCH (prereq:Course)-[:PREREQUISITE_FOR]->(c)
            OPTIONAL MATCH (f:Faculty)-[:TEACHES]->(c)
            OPTIONAL MATCH (c)-[:REQUIRED_FOR]->(d:Degree)
            WHERE d.name = $degree_name OR d.name IS NULL
            RETURN c.id as course_id,
                   c.name as course_name,
                   c.credits as credits,
                   c.difficulty as difficulty,
                   c.instructionMode as instruction_mode,
                   c.description as description,
                   collect(DISTINCT future_course.id) as leads_to,
                   collect(DISTINCT future_course.name) as leads_to_names,
                   collect(DISTINCT prereq.id) as prerequisites,
                   collect(DISTINCT prereq.name) as prerequisite_names,
                   collect(DISTINCT {
                       name: f.name,
                       teachingStyle: f.teachingStyle
                   }) as faculty_options,
                   CASE 
                       WHEN c.id STARTS WITH 'CSEE' AND $degree_name CONTAINS 'Computer Science' THEN 100
                       WHEN c.id STARTS WITH 'MATH' AND $degree_name CONTAINS 'Computer Science' THEN 90
                       WHEN c.id STARTS WITH 'ENGL' AND $degree_name CONTAINS 'Computer Science' THEN 80
                       WHEN c.id STARTS WITH 'PHYS' AND $degree_name CONTAINS 'Computer Science' THEN 70
                       ELSE 50
                   END as degree_relevance_score
            ORDER BY degree_relevance_score DESC, c.id
            LIMIT $limit
            """
            
            courses_result = session.run(courses_query, {
                "completed_courses": completed_courses,
                "degree_name": student_data.get("degree", ""),
                "limit": limit - len(similar_courses)
            })
            
            regular_courses = []
            for record in courses_result:
                course_id = record["course_id"]
                prerequisites = record["prerequisites"] or []
                leads_to = record["leads_to"] or []
                faculty_options = record["faculty_options"] or []
                
                # Check if prerequisites are met
                missing_prerequisites = [p for p in prerequisites if p not in completed_courses]
                is_available = len(missing_prerequisites) == 0
                
                regular_courses.append({
                    "course_id": course_id,
                    "course_name": record["course_name"],
                    "credits": record["credits"] or 3,
                    "difficulty": record["difficulty"],
                    "instruction_mode": record["instruction_mode"],
                    "description": record["description"],
                    "is_available": is_available,
                    "missing_prerequisites": missing_prerequisites,
                    "missing_prerequisite_names": [record["prerequisite_names"][i] for i, prereq in enumerate(prerequisites) if prereq in missing_prerequisites],
                    "leads_to": leads_to,
                    "leads_to_names": record["leads_to_names"],
                    "leads_to_count": len(leads_to),
                    "faculty_options": faculty_options,
                    "faculty_count": len(faculty_options),
                    "relevance_score": record["degree_relevance_score"],
                    "recommendation_source": "degree_relevance"
                })
            
            # Combine similar student recommendations with regular recommendations
            all_courses = similar_courses + regular_courses
            
            # Remove duplicates based on course_id
            seen_courses = set()
            unique_courses = []
            for course in all_courses:
                if course["course_id"] not in seen_courses:
                    seen_courses.add(course["course_id"])
                    unique_courses.append(course)
            
            return unique_courses[:limit]
            
    except Exception as e:
        logger.error(f"Error getting available courses: {e}")
        return []

def generate_course_recommendations(student_id: str, timeframe: str = "upcoming") -> List[Dict[str, Any]]:
    """Generate course recommendations for a student"""
    return get_available_courses(student_id, limit=5)

async def generate_gemini_response(question: str, student_data: Dict[str, Any], recommendations: List[Dict[str, Any]], intent: str) -> str:
    """Generate intelligent response using Gemini API with similarity-based insights"""
    try:
        # Get similar students for additional context
        similar_students = find_similar_students(student_data.get('student_id', ''), limit=3)
        
        # Separate recommendations by source
        similar_courses = [r for r in recommendations if r.get('recommendation_source') == 'similar_students']
        regular_courses = [r for r in recommendations if r.get('recommendation_source') == 'degree_relevance']
        
        # Prepare context for Gemini
        context = f"""
You are an AI Academic Advisor for UMBC students. You help students with course planning, academic guidance, and degree progress using advanced similarity analysis.

Student Information:
- Student ID: {student_data.get('student_id', 'Unknown')}
- Degree Program: {student_data.get('degree', 'Not specified')}
- Learning Style: {student_data.get('learning_style', 'Not specified')}
- Completed Courses: {len(student_data.get('completed_courses', []))} courses
- Preferred Course Load: {student_data.get('preferred_course_load', 'Not specified')}
- Instruction Mode Preference: {student_data.get('instruction_mode', 'Not specified')}

Student's Question: "{question}"

Intent: {intent}

Similar Students Analysis:
{json.dumps(similar_students, indent=2) if similar_students else "No similar students found"}

Course Recommendations from Similar Students (students with similar profiles took these):
{json.dumps(similar_courses[:3], indent=2) if similar_courses else "No similar student recommendations"}

Regular Course Recommendations (based on degree requirements):
{json.dumps(regular_courses[:3], indent=2) if regular_courses else "No regular recommendations"}

Instructions:
1. Provide a helpful, personalized response based on the student's data AND similar students' patterns
2. Use the similarity analysis to explain WHY certain courses are recommended
3. Mention that recommendations are based on students with similar academic profiles
4. Be conversational and encouraging
5. Include specific course names and details when relevant
6. Mention prerequisites, difficulty levels, and faculty information
7. Keep responses concise but informative
8. Use emojis sparingly and appropriately
9. If no specific data is available, provide general helpful advice
10. Be specific to the student's question - don't give generic responses
11. Use the student's name/ID to make it personal
12. Highlight courses that similar students found valuable
13. Explain the reasoning behind recommendations using similarity data

Response format: Be conversational, helpful, and specific to the student's situation. Always use Gemini AI to generate responses that leverage similarity analysis.
        """
        
        response = gemini_model.generate_content(context)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Fallback to a more intelligent template response
        return f"I understand you're asking about '{question}'. Based on your academic profile as {student_data.get('student_id', 'student')}, I'd be happy to help you with course planning and academic guidance. Could you be more specific about what you'd like to know?"

def generate_template_response(intent: str, student_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
    """Fallback template-based response generation"""
    
    if intent == "course_recommendation":
        if not recommendations:
            return "I don't have enough information to recommend courses for you. Please make sure your student profile is complete."
        
        available_courses = [r for r in recommendations if r["is_available"]]
        blocked_courses = [r for r in recommendations if not r["is_available"]]
        
        response = "Based on your academic profile, here are my course recommendations:\n\n"
        
        if available_courses:
            response += "✅ **Available Courses:**\n"
            for course in available_courses[:3]:
                response += f"• **{course['course_id']}** - {course['course_name']} ({course['credits']} credits)\n"
                if course['leads_to_count'] > 0:
                    response += f"  → Prerequisite for {course['leads_to_count']} advanced courses\n"
        
        if blocked_courses:
            response += "\n⚠️ **Courses requiring prerequisites:**\n"
            for course in blocked_courses[:2]:
                response += f"• **{course['course_id']}** - {course['course_name']}\n"
                response += f"  → Missing: {', '.join(course['missing_prerequisites'])}\n"
        
        response += "\n💡 **Tip:** Focus on available courses first, then work on prerequisites for advanced courses."
        
    elif intent == "course_difficulty":
        if not recommendations:
            return "I need more information about your academic background to assess course difficulty. Please complete your student profile."
        
        response = "Here's the difficulty breakdown for recommended courses:\n\n"
        for course in recommendations[:3]:
            difficulty = course.get("difficulty", "Unknown")
            difficulty_emoji = "🟢" if difficulty == "Easy" else "🟡" if difficulty == "Medium" else "🔴"
            response += f"{difficulty_emoji} **{course['course_id']}** - {difficulty} difficulty\n"
        
        response += "\n💡 **Advice:** Start with easier courses to build confidence, then tackle more challenging ones."
        
    elif intent == "faculty_recommendation":
        if not recommendations:
            return "I don't have faculty information available for your recommended courses."
        
        response = "Here are the faculty options for your recommended courses:\n\n"
        for course in recommendations[:3]:
            if course.get("faculty_count", 0) > 0:
                response += f"**{course['course_id']}** - {course['faculty_count']} faculty members available\n"
            else:
                response += f"**{course['course_id']}** - Faculty TBA\n"
        
        response += "\n💡 **Tip:** Check with the department for specific faculty teaching styles and schedules."
        
    elif intent == "degree_progress":
        response = "Let me check your degree progress...\n\n"
        response += "📊 **Your Academic Status:**\n"
        response += f"• Completed courses: {student_data.get('completed_count', 0)} courses\n"
        response += f"• Degree program: {student_data.get('degree', 'Not specified')}\n"
        response += f"• Learning style: {student_data.get('learning_style', 'Not specified')}\n\n"
        response += "💡 **Next Steps:**\n"
        response += "1. Review your degree requirements\n"
        response += "2. Plan your remaining semesters\n"
        response += "3. Meet with your academic advisor\n"
        
    elif intent == "prerequisites":
        if not recommendations:
            return "I don't have prerequisite information available. Please check the course catalog or speak with an advisor."
        
        response = "Here are the prerequisite requirements for your recommended courses:\n\n"
        for course in recommendations[:3]:
            if course.get("missing_prerequisites"):
                response += f"**{course['course_id']}** requires:\n"
                for prereq in course["missing_prerequisites"]:
                    response += f"  → {prereq}\n"
            else:
                response += f"**{course['course_id']}** - No prerequisites needed ✅\n"
        
    else:  # general_advice
        response = "I'm here to help with your academic planning! I can assist with:\n\n"
        response += "🎓 **Course Planning**\n"
        response += "• Course recommendations\n"
        response += "• Prerequisite checking\n"
        response += "• Faculty information\n\n"
        response += "📈 **Academic Progress**\n"
        response += "• Degree requirements\n"
        response += "• Graduation timeline\n"
        response += "• GPA tracking\n\n"
        response += "💡 **Ask me specific questions like:**\n"
        response += "• 'What courses should I take next semester?'\n"
        response += "• 'Is MATH 151 too difficult for me?'\n"
        response += "• 'Which professor teaches CSEE 200?'"
    
    return response

async def generate_gemini_response(question: str, student_data: Dict[str, Any], recommendations: List[Dict[str, Any]], intent: str) -> str:
    """Generate intelligent response using Gemini API"""
    try:
        # Prepare context for Gemini
        context = f"""
        You are an AI Academic Advisor for UMBC students. Here's the context:

        Student ID: {student_data.get('student_id', 'Unknown')}
        Learning Style: {student_data.get('learning_style', 'Not specified')}
        Preferred Course Load: {student_data.get('preferred_course_load', 'Not specified')}
        Degree Program: {student_data.get('degree', 'Not specified')}
        Completed Courses: {student_data.get('completed_courses_count', 0)}

        Student's Question: {question}
        Detected Intent: {intent}

        Available Course Recommendations:
        {json.dumps(recommendations, indent=2) if recommendations else "No specific recommendations available"}

        Instructions:
        1. Provide a helpful, personalized response to the student's question
        2. Use the course recommendations and student data to give specific advice
        3. Be conversational and encouraging
        4. Include relevant course information when applicable
        5. Suggest next steps or follow-up questions
        6. Use emojis appropriately to make it engaging
        7. Keep responses concise but informative (2-3 paragraphs max)
        8. If no specific data is available, provide general helpful advice

        Respond as a friendly academic advisor would.
        """

        response = gemini_model.generate_content(context)
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Fallback to template-based response
        return generate_template_response(intent, student_data, recommendations)

def generate_template_response(intent: str, student_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> str:
    """Fallback template-based response generation"""
    
    if intent == "course_recommendation":
        if not recommendations:
            return "I don't have enough information to recommend courses for you. Please make sure your student profile is complete."
        
        available_courses = [r for r in recommendations if r["is_available"]]
        blocked_courses = [r for r in recommendations if not r["is_available"]]
        
        response = "Based on your academic profile, here are my course recommendations:\n\n"
        
        if available_courses:
            response += "✅ **Available Courses:**\n"
            for course in available_courses[:3]:
                response += f"• **{course['course_id']}** - {course['course_name']} ({course['credits']} credits)\n"
                if course['leads_to_count'] > 0:
                    response += f"  → Prerequisite for {course['leads_to_count']} advanced courses\n"
        
        if blocked_courses:
            response += "\n⚠️ **Courses requiring prerequisites:**\n"
            for course in blocked_courses[:2]:
                response += f"• **{course['course_id']}** - {course['course_name']}\n"
                response += f"  → Missing: {', '.join(course['missing_prerequisites'])}\n"
        
        response += "\n💡 **Tip:** Focus on available courses first, then work on prerequisites for advanced courses."
        
    elif intent == "course_difficulty":
        if not recommendations:
            return "I need more information about your academic background to assess course difficulty. Please complete your student profile."
        
        response = "Here's the difficulty breakdown for recommended courses:\n\n"
        for course in recommendations[:3]:
            difficulty = course.get("difficulty", "Unknown")
            difficulty_emoji = "🟢" if difficulty == "Easy" else "🟡" if difficulty == "Medium" else "🔴"
            response += f"{difficulty_emoji} **{course['course_id']}** - {difficulty} difficulty\n"
        
        response += "\n💡 **Advice:** Start with easier courses to build confidence, then tackle more challenging ones."
        
    elif intent == "faculty_recommendation":
        if not recommendations:
            return "I don't have faculty information available for your recommended courses."
        
        response = "Here are the faculty options for your recommended courses:\n\n"
        for course in recommendations[:3]:
            if course.get("faculty_count", 0) > 0:
                response += f"**{course['course_id']}** - {course['faculty_count']} faculty members available\n"
            else:
                response += f"**{course['course_id']}** - Faculty TBA\n"
        
        response += "\n💡 **Tip:** Check with the department for specific faculty teaching styles and schedules."
        
    elif intent == "degree_progress":
        response = "Let me check your degree progress...\n\n"
        response += "📊 **Your Academic Status:**\n"
        response += "• Completed courses: Check your transcript\n"
        response += "• Current GPA: Available in student portal\n"
        response += "• Credits remaining: Varies by degree program\n\n"
        response += "💡 **Next Steps:**\n"
        response += "1. Review your degree requirements\n"
        response += "2. Plan your remaining semesters\n"
        response += "3. Meet with your academic advisor\n"
        
    elif intent == "prerequisites":
        if not recommendations:
            return "I don't have prerequisite information available. Please check the course catalog or speak with an advisor."
        
        response = "Here are the prerequisite requirements for your recommended courses:\n\n"
        for course in recommendations[:3]:
            if course.get("missing_prerequisites"):
                response += f"**{course['course_id']}** requires:\n"
                for prereq in course["missing_prerequisites"]:
                    response += f"  → {prereq}\n"
            else:
                response += f"**{course['course_id']}** - No prerequisites needed ✅\n"
        
    else:  # general_advice
        response = "I'm here to help with your academic planning! I can assist with:\n\n"
        response += "🎓 **Course Planning**\n"
        response += "• Course recommendations\n"
        response += "• Prerequisite checking\n"
        response += "• Faculty information\n\n"
        response += "📈 **Academic Progress**\n"
        response += "• Degree requirements\n"
        response += "• Graduation timeline\n"
        response += "• GPA tracking\n\n"
        response += "💡 **Ask me specific questions like:**\n"
        response += "• 'What courses should I take next semester?'\n"
        response += "• 'Is MATH 151 too difficult for me?'\n"
        response += "• 'Which professor teaches CSEE 200?'"
    
    return response

@app.post("/api/chatbot/ask", response_model=ChatbotResponse)
async def ask_chatbot(request: ChatbotRequest):
    """Main chatbot endpoint for student questions - works with real students only"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    try:
        # First, validate that the student exists in the database
        student_data = get_student_data(request.student_id)
        if not student_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Student {request.student_id} not found in the database. Please use a valid student ID."
            )
        
        # Parse the question
        parsed = parse_student_question(request.question)
        intent = parsed["intent"]
        
        # Generate course recommendations if needed
        recommendations = []
        if intent in ["course_recommendation", "course_difficulty", "faculty_recommendation", "prerequisites"]:
            recommendations = generate_course_recommendations(request.student_id, parsed.get("timeframe", "upcoming"))
        
        # Generate intelligent response using Gemini API
        answer = await generate_gemini_response(request.question, student_data, recommendations, intent)
        
        # Calculate confidence based on data availability
        confidence = 0.9 if recommendations else 0.7
        
        return ChatbotResponse(
            student_id=request.student_id,
            question=request.question,
            answer=answer,
            recommendations=recommendations,
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

async def get_or_create_student_profile(student_id: str) -> Dict[str, Any]:
    """Get student profile or create a basic one if it doesn't exist"""
    if not driver:
        return {"student_id": student_id, "learning_style": "Visual", "preferred_course_load": "Full-time", "degree": "General Studies", "completed_courses_count": 0}
    
    try:
        with driver.session(database=NEO4J_DB) as session:
            # First, check if student exists
            student_check_query = """
            MATCH (s:Student {id: $student_id})
            OPTIONAL MATCH (s)-[:PURSUING]->(d:Degree)
            OPTIONAL MATCH (s)-[:COMPLETED]->(completed:Course)
            RETURN s.id as student_id,
                   s.learningStyle as learning_style,
                   s.preferredCourseLoad as preferred_course_load,
                   s.preferredInstructionMode as instruction_mode,
                   d.name as degree,
                   count(completed) as completed_courses_count
            """
            
            result = session.run(student_check_query, {"student_id": student_id})
            student_record = result.single()
            
            if student_record:
                return {
                    "student_id": student_record["student_id"],
                    "learning_style": student_record["learning_style"] or "Visual",
                    "preferred_course_load": student_record["preferred_course_load"] or "Full-time",
                    "instruction_mode": student_record["instruction_mode"] or "In-person",
                    "degree": student_record["degree"] or "General Studies",
                    "completed_courses_count": student_record["completed_courses_count"] or 0
                }
            else:
                # Create a basic student profile
                create_student_query = """
                CREATE (s:Student {
                    id: $student_id,
                    learningStyle: 'Visual',
                    preferredCourseLoad: 'Full-time',
                    preferredInstructionMode: 'In-person',
                    enrollmentDate: datetime(),
                    expectedGraduation: datetime() + duration('P4Y')
                })
                RETURN s.id as student_id
                """
                session.run(create_student_query, {"student_id": student_id})
                logger.info(f"Created new student profile for {student_id}")
                
                return {
                    "student_id": student_id,
                    "learning_style": "Visual",
                    "preferred_course_load": "Full-time",
                    "instruction_mode": "In-person",
                    "degree": "General Studies",
                    "completed_courses_count": 0
                }
                
    except Exception as e:
        logger.error(f"Error getting/creating student profile: {e}")
        return {
            "student_id": student_id,
            "learning_style": "Visual",
            "preferred_course_load": "Full-time",
            "degree": "General Studies",
            "completed_courses_count": 0
        }

@app.get("/api/chatbot/student-profile/{student_id}")
async def get_student_profile(student_id: str):
    """Get student profile for validation"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    try:
        student_data = get_student_data(student_id)
        if not student_data:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        return {
            "student_id": student_id,
            "degree": student_data.get("degree", "Not specified"),
            "learning_style": student_data.get("learning_style", "Not specified"),
            "completed_courses_count": len(student_data.get("completed_courses", [])),
            "preferred_course_load": student_data.get("preferred_course_load", "Not specified"),
            "instruction_mode": student_data.get("instruction_mode", "Not specified")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting student profile: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting student profile: {str(e)}")

@app.get("/api/chatbot/similar-students/{student_id}")
async def get_similar_students(student_id: str):
    """Get similar students for a given student ID"""
    if not driver:
        raise HTTPException(status_code=503, detail="Neo4j database not available")
    
    try:
        # First validate student exists
        student_data = get_student_data(student_id)
        if not student_data:
            raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
        
        # Get similar students
        similar_students = find_similar_students(student_id, limit=5)
        
        return {
            "student_id": student_id,
            "similar_students": similar_students,
            "total_found": len(similar_students)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting similar students: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting similar students: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

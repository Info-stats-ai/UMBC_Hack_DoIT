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

# Load model
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
        model = None
        feature_names = []
        logger.warning("No model found! Please train the model first.")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

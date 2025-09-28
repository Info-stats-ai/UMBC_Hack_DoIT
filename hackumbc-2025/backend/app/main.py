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
NEO4J_PASSWORD = "Iwin@27100"
NEO4J_DB = "smalldata"

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
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    logger.info("Connected to Neo4j database")
except Exception as e:
    driver = None
    logger.error(f"Failed to connect to Neo4j: {e}")

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

# Helper functions
def get_student_course_features(student_id: str, course_id: str) -> pd.DataFrame:
    """Extract features for a student-course pair from Neo4j"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# UMBC Student Success Dashboard 🎓

A comprehensive AI-powered academic success platform designed to help UMBC students navigate their academic journey with personalized insights, course recommendations, and risk assessment.

## 📊 Project Overview

This project is a full-stack web application that combines **Neo4j graph database**, **machine learning models**, and **modern web technologies** to create an intelligent academic advisory system. The platform provides students with personalized course recommendations, academic risk assessment, study group matching, mentorship connections, and progress tracking.

## 🎯 What We Built

### Core Features

1. **🎯 Risk Assessment Dashboard**
   - AI-powered academic risk prediction for specific courses
   - Personalized risk scores based on student profile and course history
   - Visual risk indicators and recommendations

2. **📚 Course Planning Assistant**
   - Intelligent course recommendations based on learning style and academic history
   - Prerequisite tracking and course sequencing
   - Term-based course scheduling

3. **👥 Study Groups & Mentorship**
   - Smart study group matching based on learning styles and course preferences
   - AI-powered mentorship pairing system
   - Collaborative learning platform

4. **📈 Progress Tracking**
   - Academic performance monitoring
   - Goal setting and achievement tracking
   - Visual progress analytics

5. **🤖 AI Advisory Chatbot**
   - Interactive AI assistant for academic guidance
   - Real-time course and degree advice
   - Personalized recommendations

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Neo4j** - Graph database for relationship modeling
- **Scikit-learn** - Machine learning models
- **LightGBM** - Gradient boosting for risk prediction
- **Google Generative AI** - AI chatbot integration

### Frontend
- **HTML5/CSS3** - Modern responsive design
- **JavaScript (ES6+)** - Interactive user interface
- **CSS Animations** - Engaging user experience
- **Glass Morphism Design** - Modern UI aesthetics

### Data & ML
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Optuna** - Hyperparameter optimization
- **Joblib** - Model serialization

## 📈 Data Architecture

### Synthetic Dataset
We generated a comprehensive synthetic dataset representing UMBC's academic ecosystem:

- **500 Students** with diverse learning styles and academic profiles
- **100 Courses** across Computer Science and Business departments
- **30 Faculty Members** with teaching assignments
- **4 Degree Programs** (BS/BA in CS and Business)
- **1,068 Risk Relationships** between students and courses

### Graph Database Schema
```
Students ←→ Courses (COMPLETED, ENROLLED_IN)
Students ←→ Students (SIMILAR_LEARNING_STYLE, SIMILAR_PERFORMANCE)
Courses ←→ Courses (PREREQUISITE_FOR, LEADS_TO, SIMILAR_CONTENT)
Students ←→ Degrees (PURSUING)
Faculty ←→ Courses (TEACHES)
```

### Machine Learning Models
- **Academic Risk Prediction**: Multiclass classification (A, B, C, D, F grades)
- **Feature Engineering**: 50+ features including learning style, course difficulty, prerequisites
- **Model Performance**: 85%+ accuracy on test data

## 🚀 Why We Built This

### Problem Statement
Traditional academic advising systems are often:
- **Generic** - One-size-fits-all recommendations
- **Reactive** - Only help after problems occur
- **Isolated** - Don't leverage peer learning opportunities
- **Manual** - Require extensive advisor time

### Our Solution
We created an **intelligent, proactive, and personalized** system that:
- **Predicts** academic risks before they become problems
- **Personalizes** recommendations based on individual learning styles
- **Connects** students with similar academic profiles
- **Automates** routine advising tasks

## 🎯 What We Achieved

### Technical Achievements
1. **Graph Database Integration**: Successfully modeled complex academic relationships in Neo4j
2. **ML Pipeline**: Built end-to-end machine learning pipeline for risk prediction
3. **Real-time API**: Created responsive REST API with FastAPI
4. **Modern UI**: Developed engaging, animated user interface
5. **AI Integration**: Implemented conversational AI for student guidance

### Impact Metrics
- **1,068 Risk Relationships** generated and analyzed
- **85%+ Model Accuracy** for academic risk prediction
- **500 Students** with personalized profiles
- **100 Courses** with difficulty and similarity analysis
- **Real-time Recommendations** based on graph analytics

### Key Innovations
1. **Graph-based Similarity**: Used Neo4j to find students with similar learning patterns
2. **Multi-modal Risk Assessment**: Combined academic history, learning style, and course difficulty
3. **Dynamic Course Sequencing**: Intelligent prerequisite-aware course planning
4. **Collaborative Learning**: AI-powered study group and mentorship matching

## 📁 Project Structure

```
hackumbc-2025/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Main API server
│   │   ├── mentorship_service.py
│   │   └── study_groups_service.py
│   └── requirements.txt
├── frontend/               # Web interface
│   ├── index.html         # Main dashboard
│   ├── course-planning.html
│   ├── risk-assessment.html
│   ├── study-groups.html
│   ├── mentorship.html
│   └── progress-tracking.html
├── ml/                     # Machine learning
│   ├── data/              # Processed datasets
│   ├── models/            # Trained ML models
│   └── notebooks/         # Jupyter notebooks
├── umbc_data/             # Neo4j data
│   ├── cypher/           # Database scripts
│   └── csv/              # Data files
└── create_risk_relationships.py  # Risk data generator
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ 
- Neo4j Database
- Node.js (for frontend development)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd hackumbc-2025
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up Neo4j Database**
```bash
# Start Neo4j server
# Import data using the cypher scripts in umbc_data/cypher/
```

4. **Generate Risk Relationships**
```bash
python create_risk_relationships.py
```

5. **Start the Backend Server**
```bash
cd backend
python run_server.py
```

6. **Access the Application**
Open your browser and navigate to `http://localhost:8000`

## 🔧 Configuration

### Environment Variables
- `NEO4J_URI`: Neo4j database connection string
- `NEO4J_USER`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password
- `GOOGLE_API_KEY`: Google Generative AI API key

### Database Setup
1. Import the cypher scripts in `umbc_data/cypher/` in numerical order
2. Load the risk relationships using `create_comprehensive_risk_relationships.cypher`

## 📊 Data Insights

### Risk Distribution
- **MEDIUM Risk**: 84% (897 relationships)
- **LOW Risk**: 9.6% (103 relationships)  
- **HIGH Risk**: 6.4% (68 relationships)

### Top Risky Courses
1. CSDD 100: 42.6% average risk
2. BWWW 200: 41.2% average risk
3. CSGG 300: 41.1% average risk

### Learning Style Distribution
- **Visual**: 35%
- **Auditory**: 25%
- **Kinesthetic**: 20%
- **Reading-Writing**: 20%

## 🎨 UI/UX Features

- **Glass Morphism Design**: Modern, translucent interface elements
- **Animated Backgrounds**: Dynamic gradient backgrounds with floating orbs
- **Interactive Cards**: Hover effects and smooth transitions
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Loading Skeletons**: Smooth loading states for better UX

## 🔮 Future Enhancements

1. **Real-time Notifications**: Push notifications for important academic deadlines
2. **Mobile App**: Native iOS/Android applications
3. **Advanced Analytics**: More sophisticated ML models and visualizations
4. **Integration**: Connect with existing UMBC systems (Blackboard, etc.)
5. **Gamification**: Achievement badges and progress rewards

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---

**Built with ❤️ for UMBC Students**

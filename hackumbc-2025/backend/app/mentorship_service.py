"""
Mentorship Matching Service - Connect students with mentors based on academic paths and interests
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import logging
from dataclasses import dataclass
from collections import defaultdict
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class MentorProfile:
    """Represents a mentor profile"""
    mentor_id: str
    name: str
    mentor_type: str  # senior_student, faculty, industry_professional
    department: str
    major: str
    year_level: int  # For students: 1-4, For faculty/industry: years of experience
    research_interests: List[str]
    industry_experience: List[str]
    skills: List[str]
    availability: str
    contact_info: str
    bio: str
    mentoring_capacity: int
    current_mentees: int
    rating: float
    specializations: List[str]

@dataclass
class MenteeProfile:
    """Represents a mentee profile"""
    mentee_id: str
    name: str
    major: str
    year_level: int
    academic_interests: List[str]
    career_goals: List[str]
    skills_to_develop: List[str]
    preferred_mentor_type: List[str]
    learning_style: str
    availability: str
    goals: str

@dataclass
class MentorshipMatch:
    """Represents a mentorship match"""
    match_id: str
    mentor: MentorProfile
    mentee: MenteeProfile
    compatibility_score: float
    compatibility_factors: Dict[str, float]
    match_type: str
    recommended_meeting_frequency: str
    suggested_activities: List[str]
    goals: List[str]

class MentorshipService:
    """Service for mentorship matching and management"""
    
    def __init__(self, neo4j_driver=None, csv_data_path="../../ml/notebooks/umbc_data/csv"):
        self.driver = neo4j_driver
        self.csv_path = csv_data_path
        self.students_df = None
        self.faculty_df = None
        self.courses_df = None
        self.completed_df = None
        
        # Mock data for demonstration - in production, this would come from databases
        self._load_csv_data()
        self._generate_mentor_data()
    
    def _load_csv_data(self):
        """Load existing CSV data"""
        try:
            base_path = os.path.join(os.path.dirname(__file__), self.csv_path)
            
            # Load student data
            students_path = os.path.join(base_path, "students.csv")
            if os.path.exists(students_path):
                self.students_df = pd.read_csv(students_path)
                logger.info(f"Loaded {len(self.students_df)} students for mentorship")
            
            # Load faculty data
            faculty_path = os.path.join(base_path, "faculty.csv")
            if os.path.exists(faculty_path):
                self.faculty_df = pd.read_csv(faculty_path)
                logger.info(f"Loaded {len(self.faculty_df)} faculty for mentorship")
            
            # Load courses for research interests mapping
            courses_path = os.path.join(base_path, "courses.csv")
            if os.path.exists(courses_path):
                self.courses_df = pd.read_csv(courses_path)
                logger.info(f"Loaded {len(self.courses_df)} courses for research mapping")
            
            # Load completed courses for academic path analysis
            completed_path = os.path.join(base_path, "completed_courses.csv")
            if os.path.exists(completed_path):
                self.completed_df = pd.read_csv(completed_path)
                logger.info(f"Loaded {len(self.completed_df)} completed course records")
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
    
    def _generate_mentor_data(self):
        """Generate mentor profiles from real data relationships"""
        
        # Load additional CSV files for mentorship relationships
        try:
            base_path = os.path.join(os.path.dirname(__file__), self.csv_path)
            
            # Load degree data to understand student majors
            degree_path = os.path.join(base_path, "student_degree.csv")
            self.degree_df = None
            if os.path.exists(degree_path):
                self.degree_df = pd.read_csv(degree_path)
                logger.info(f"Loaded {len(self.degree_df)} degree relationships")
            
            # Load teaching relationships
            teaching_path = os.path.join(base_path, "teaching.csv")
            teaching_df = None
            if os.path.exists(teaching_path):
                teaching_df = pd.read_csv(teaching_path)
                logger.info(f"Loaded {len(teaching_df)} teaching relationships")
            
            # Load performance similarity for peer mentoring (reuse already loaded data)
            if hasattr(self, 'performance_similarity_df') and self.performance_similarity_df is not None:
                performance_df = self.performance_similarity_df
            else:
                performance_path = os.path.join(base_path, "performance_similarity.csv")
                performance_df = None
                if os.path.exists(performance_path):
                    performance_df = pd.read_csv(performance_path)
                    self.performance_similarity_df = performance_df
                    logger.info(f"Loaded {len(performance_df)} performance similarities")
                
        except Exception as e:
            logger.error(f"Error loading additional mentor data: {e}")
            self.degree_df = None
            teaching_df = None
            performance_df = None
        
        # Generate senior student mentors (students who can mentor others)
        self.senior_mentors = []
        if self.students_df is not None and self.completed_df is not None:
            # Find students who have completed many courses (potential senior mentors)
            course_counts = self.completed_df.groupby(':START_ID(Student)').size()
            senior_candidates = course_counts[course_counts >= course_counts.quantile(0.7)].index.tolist()
            
            for student_id in senior_candidates[:40]:  # Limit to top 40 senior students
                student_info = self.students_df[self.students_df['id:ID(Student)'] == student_id]
                if student_info.empty:
                    continue
                
                student_row = student_info.iloc[0]
                
                # Get student's major from degree relationships
                major = self._get_student_major(student_id, self.degree_df)
                
                # Get courses completed and derive experience areas
                completed_courses = self._get_student_courses(student_id)
                experience_areas = self._derive_experience_from_courses(completed_courses)
                
                mentor = MentorProfile(
                    mentor_id=student_id,
                    name=student_row['name'],
                    mentor_type="senior_student",
                    department=major.split('-')[0] if '-' in major else major,
                    major=major,
                    year_level=self._calculate_year_level(student_id),
                    research_interests=experience_areas,
                    industry_experience=[],
                    skills=self._derive_skills_from_courses(completed_courses),
                    availability=self._map_availability(student_row.get('workHoursPerWeek:int', 0)),
                    contact_info=f"{student_row['name'].lower().replace(' ', '.')}@umbc.edu",
                    bio=f"Senior student in {major}. Has completed {len(completed_courses)} courses and can help with academic planning.",
                    mentoring_capacity=min(3, max(1, 4 - int(student_row.get('workHoursPerWeek:int', 0) / 15))),
                    current_mentees=0,  # Start with 0, would be updated in real system
                    rating=self._calculate_peer_rating(student_id),
                    specializations=experience_areas[:2]  # Top 2 areas
                )
                self.senior_mentors.append(mentor)
        
        # Generate faculty mentors from real faculty data
        self.faculty_mentors = []
        if self.faculty_df is not None:
            for _, faculty in self.faculty_df.iterrows():
                # Get courses this faculty teaches
                faculty_courses = []
                if teaching_df is not None:
                    faculty_teaching = teaching_df[teaching_df[':START_ID(Faculty)'] == faculty['id:ID(Faculty)']]
                    faculty_courses = faculty_teaching[':END_ID(Course)'].tolist()
                
                # Derive research interests from courses taught
                research_interests = self._derive_faculty_interests(faculty_courses, faculty['department'])
                
                mentor = MentorProfile(
                    mentor_id=faculty['id:ID(Faculty)'],
                    name=faculty['name'],
                    mentor_type="faculty",
                    department=faculty['department'],
                    major=faculty['department'],
                    year_level=self._estimate_faculty_experience(faculty['avgRating:float']),
                    research_interests=research_interests,
                    industry_experience=[faculty['department']],  # Academic experience
                    skills=self._derive_faculty_skills(faculty['teachingStyle'], faculty['department']),
                    availability="Office Hours",
                    contact_info=f"{faculty['name'].lower().replace(' ', '.')}@umbc.edu",
                    bio=f"Professor in {faculty['department']} specializing in {', '.join(research_interests[:2])}. Teaching style: {faculty['teachingStyle']}.",
                    mentoring_capacity=max(2, min(6, int(faculty['avgRating:float']))),
                    current_mentees=0,
                    rating=faculty['avgRating:float'],
                    specializations=research_interests
                )
                self.faculty_mentors.append(mentor)
        
        # For now, no industry mentors since they're not in the data
        # This could be extended with additional data sources
        self.industry_mentors = []
        
        logger.info(f"Generated {len(self.senior_mentors)} senior mentors, {len(self.faculty_mentors)} faculty mentors from real data")
    
    def _get_department_from_courses(self, student_id: str) -> str:
        """Determine department based on completed courses"""
        if self.completed_df is not None:
            student_courses = self.completed_df[self.completed_df[':START_ID(Student)'] == student_id]
            if not student_courses.empty:
                # Simple heuristic: most common course prefix
                course_prefixes = student_courses[':END_ID(Course)'].str[:2].value_counts()
                if not course_prefixes.empty:
                    prefix = course_prefixes.index[0]
                    dept_mapping = {
                        'CS': 'Computer Science', 'MA': 'Mathematics', 'BI': 'Biology',
                        'CH': 'Chemistry', 'PH': 'Physics', 'EN': 'English',
                        'HI': 'History', 'PS': 'Psychology', 'EC': 'Economics'
                    }
                    return dept_mapping.get(prefix, "Computer Science")
        return "Computer Science"
    
    def _get_major_from_courses(self, student_id: str) -> str:
        """Determine major based on completed courses"""
        return self._get_department_from_courses(student_id)
    
    def _get_student_major(self, student_id: str, degree_df) -> str:
        """Get student's major from degree relationships"""
        if degree_df is not None:
            student_degrees = degree_df[degree_df[':START_ID(Student)'] == student_id]
            if not student_degrees.empty:
                # Return the first degree found
                return student_degrees.iloc[0][':END_ID(Degree)']
        return "Computer Science"  # Default fallback
    
    def _get_student_courses(self, student_id: str) -> List[str]:
        """Get list of courses completed by student"""
        if self.completed_df is not None:
            student_courses = self.completed_df[self.completed_df[':START_ID(Student)'] == student_id]
            return student_courses[':END_ID(Course)'].tolist()
        return []
    
    def _derive_experience_from_courses(self, courses: List[str]) -> List[str]:
        """Derive experience areas from completed courses"""
        areas = []
        course_prefixes = {}
        
        for course in courses:
            if course and len(course) >= 2:
                prefix = course[:2]
                course_prefixes[prefix] = course_prefixes.get(prefix, 0) + 1
        
        # Map course prefixes to experience areas
        prefix_mapping = {
            'CS': ['Computer Science', 'Programming', 'Software Development'],
            'MA': ['Mathematics', 'Statistics', 'Data Analysis'],
            'BI': ['Biology', 'Life Sciences', 'Research'],
            'CH': ['Chemistry', 'Laboratory Skills', 'Scientific Research'],
            'PH': ['Physics', 'Problem Solving', 'Mathematical Modeling'],
            'EN': ['English', 'Communication', 'Writing'],
            'HI': ['History', 'Research', 'Critical Thinking'],
            'PS': ['Psychology', 'Human Behavior', 'Research Methods'],
            'EC': ['Economics', 'Data Analysis', 'Business']
        }
        
        # Get top course areas and map to experience
        sorted_prefixes = sorted(course_prefixes.items(), key=lambda x: x[1], reverse=True)
        for prefix, count in sorted_prefixes[:3]:  # Top 3 areas
            if prefix in prefix_mapping:
                areas.extend(prefix_mapping[prefix])
        
        return list(set(areas))[:5]  # Return unique areas, max 5
    
    def _derive_skills_from_courses(self, courses: List[str]) -> List[str]:
        """Derive skills from completed courses"""
        skills = set()
        
        for course in courses:
            if course:
                # Basic skill derivation from course prefixes
                if course.startswith('CS'):
                    skills.update(['Programming', 'Problem Solving', 'Software Development'])
                elif course.startswith('MA'):
                    skills.update(['Mathematics', 'Analytical Thinking', 'Statistics'])
                elif course.startswith('EN'):
                    skills.update(['Writing', 'Communication', 'Critical Reading'])
                elif course.startswith('BI'):
                    skills.update(['Research', 'Laboratory Skills', 'Scientific Method'])
                
                # Add general academic skills
                skills.update(['Study Skills', 'Time Management'])
        
        return list(skills)[:8]  # Max 8 skills
    
    def _map_availability(self, work_hours: int) -> str:
        """Map work hours to availability"""
        if work_hours == 0:
            return "Flexible"
        elif work_hours < 20:
            return "Weekdays"
        elif work_hours < 30:
            return "Evenings"
        else:
            return "Weekends"
    
    def _calculate_year_level(self, student_id: str) -> int:
        """Calculate student year level based on courses completed"""
        if self.completed_df is not None:
            course_count = len(self.completed_df[self.completed_df[':START_ID(Student)'] == student_id])
            # Rough estimation: 8 courses per year
            return min(4, max(1, course_count // 8 + 1))
        return 3  # Default to junior
    
    def _calculate_peer_rating(self, student_id: str) -> float:
        """Calculate peer mentor rating based on academic performance"""
        if self.completed_df is not None:
            student_courses = self.completed_df[self.completed_df[':START_ID(Student)'] == student_id]
            if not student_courses.empty:
                # Convert grades to GPA approximation
                grade_mapping = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D': 1.0, 'F': 0.0}
                grades = student_courses['grade'].tolist()
                gpa_sum = sum(grade_mapping.get(grade, 2.5) for grade in grades)
                avg_gpa = gpa_sum / len(grades) if grades else 2.5
                # Convert GPA to 5-point rating scale
                return min(5.0, max(3.0, avg_gpa + 1.0))
        return 4.0  # Default rating
    
    def _derive_faculty_interests(self, courses: List[str], department: str) -> List[str]:
        """Derive faculty research interests from courses taught and department"""
        interests = []
        
        # Department-based research areas
        dept_interests = {
            'Computer Science': ['Software Engineering', 'Algorithms', 'Data Structures', 'Programming Languages', 'Systems'],
            'Biology': ['Molecular Biology', 'Genetics', 'Ecology', 'Cell Biology', 'Bioinformatics'],
            'Mathematics': ['Algebra', 'Calculus', 'Statistics', 'Mathematical Modeling', 'Applied Mathematics'],
            'Chemistry': ['Organic Chemistry', 'Physical Chemistry', 'Analytical Chemistry', 'Biochemistry'],
            'Physics': ['Theoretical Physics', 'Experimental Physics', 'Quantum Mechanics', 'Thermodynamics'],
            'Psychology': ['Cognitive Psychology', 'Behavioral Psychology', 'Research Methods', 'Statistics']
        }
        
        if department in dept_interests:
            interests.extend(dept_interests[department])
        
        # Course-specific interests
        for course in courses:
            if course and course.startswith('CS'):
                if '100' in course or '200' in course:
                    interests.append('Introductory Programming')
                elif '300' in course or '400' in course:
                    interests.append('Advanced Computer Science')
        
        return list(set(interests))[:6]  # Max 6 interests
    
    def _derive_faculty_skills(self, teaching_style: str, department: str) -> List[str]:
        """Derive faculty skills from teaching style and department"""
        skills = ['Teaching', 'Research', 'Academic Mentoring']
        
        if teaching_style:
            styles = teaching_style.split(';')
            for style in styles:
                if 'Problem-Based' in style:
                    skills.append('Problem-Solving Guidance')
                elif 'Research-Oriented' in style:
                    skills.append('Research Supervision')
                elif 'Hands-on' in style:
                    skills.append('Practical Skills Training')
                elif 'Socratic' in style:
                    skills.append('Critical Thinking Development')
        
        # Department-specific skills
        if department == 'Computer Science':
            skills.extend(['Programming', 'Software Development', 'Technical Skills'])
        elif department == 'Mathematics':
            skills.extend(['Mathematical Analysis', 'Statistical Methods'])
        elif department == 'Biology':
            skills.extend(['Laboratory Techniques', 'Scientific Research'])
        
        return list(set(skills))[:8]  # Max 8 skills
    
    def _estimate_faculty_experience(self, rating: float) -> int:
        """Estimate faculty experience years based on rating and other factors"""
        # Higher rated faculty likely have more experience
        base_years = max(5, int(rating * 4))  # 5-20 years based on rating
        return min(25, base_years)
    
    def find_mentors(self, mentee_request: Dict[str, Any]) -> List[MentorshipMatch]:
        """Find compatible mentors for a mentee"""
        
        try:
            # Create mentee profile from request
            mentee = MenteeProfile(
                mentee_id=mentee_request['mentee_id'],
                name=mentee_request.get('name', 'Student'),
                major=mentee_request.get('major', 'Computer Science'),
                year_level=mentee_request.get('year_level', 1),
                academic_interests=mentee_request.get('academic_interests', []),
                career_goals=mentee_request.get('career_goals', []),
                skills_to_develop=mentee_request.get('skills_to_develop', []),
                preferred_mentor_type=mentee_request.get('preferred_mentor_type', ['senior_student', 'faculty']),
                learning_style=mentee_request.get('learning_style', 'Visual'),
                availability=mentee_request.get('availability', 'Flexible'),
                goals=mentee_request.get('goals', 'Academic and career guidance')
            )
            
            # Get all available mentors
            all_mentors = []
            
            if 'senior_student' in mentee.preferred_mentor_type:
                all_mentors.extend(self.senior_mentors)
            if 'faculty' in mentee.preferred_mentor_type:
                all_mentors.extend(self.faculty_mentors)
            
            # Filter available mentors (those with capacity)
            available_mentors = [m for m in all_mentors if m.current_mentees < m.mentoring_capacity]
            
            # Calculate compatibility scores
            matches = []
            for mentor in available_mentors:
                compatibility_score, factors = self._calculate_mentor_compatibility(mentee, mentor)
                
                if compatibility_score > 0.3:  # Minimum threshold
                    match = MentorshipMatch(
                        match_id=f"MATCH_{mentee.mentee_id}_{mentor.mentor_id}",
                        mentor=mentor,
                        mentee=mentee,
                        compatibility_score=compatibility_score,
                        compatibility_factors=factors,
                        match_type=mentor.mentor_type,
                        recommended_meeting_frequency=self._suggest_meeting_frequency(mentor.mentor_type),
                        suggested_activities=self._suggest_activities(mentee, mentor),
                        goals=self._suggest_goals(mentee, mentor)
                    )
                    matches.append(match)
            
            # Sort by compatibility score
            matches.sort(key=lambda x: x.compatibility_score, reverse=True)
            
            return matches[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Error finding mentors: {e}")
            return []
    
    def _calculate_mentor_compatibility(self, mentee: MenteeProfile, mentor: MentorProfile) -> Tuple[float, Dict[str, float]]:
        """Calculate compatibility score between mentee and mentor using real data relationships"""
        
        factors = {}
        
        # Academic interest alignment
        interest_score = self._calculate_interest_overlap(
            mentee.academic_interests, mentor.research_interests + mentor.specializations
        )
        factors['academic_interests'] = interest_score
        
        # Major/department compatibility - stronger emphasis on real degree relationships
        major_score = self._calculate_major_compatibility(mentee.major, mentor.major, mentor.department)
        factors['major_alignment'] = major_score
        
        # Career goals vs mentor experience
        career_score = self._calculate_career_alignment(mentee.career_goals, mentor.industry_experience)
        factors['career_alignment'] = career_score
        
        # Skills development potential
        skills_score = self._calculate_skills_overlap(mentee.skills_to_develop, mentor.skills)
        factors['skills_development'] = skills_score
        
        # Learning style compatibility (if we have performance similarity data)
        learning_score = self._calculate_learning_compatibility(mentee.mentee_id, mentor.mentor_id)
        factors['learning_compatibility'] = learning_score
        
        # Availability compatibility
        availability_score = self._calculate_availability_compatibility(mentee.availability, mentor.availability)
        factors['availability'] = availability_score
        
        # Mentor type preference
        type_score = 1.0 if mentor.mentor_type in mentee.preferred_mentor_type else 0.3
        factors['mentor_type'] = type_score
        
        # Experience level appropriateness
        experience_score = self._calculate_experience_appropriateness(mentee.year_level, mentor.year_level, mentor.mentor_type)
        factors['experience_level'] = experience_score
        
        # Mentor rating and capacity
        quality_score = min(mentor.rating / 5.0, 1.0) * (1.0 - mentor.current_mentees / mentor.mentoring_capacity)
        factors['mentor_quality'] = quality_score
        
        # Academic performance compatibility (for peer mentors)
        performance_score = self._calculate_performance_compatibility(mentee.mentee_id, mentor.mentor_id)
        factors['performance_compatibility'] = performance_score
        
        # Weighted overall score
        weights = {
            'academic_interests': 0.20,
            'major_alignment': 0.20,
            'career_alignment': 0.15,
            'skills_development': 0.15,
            'learning_compatibility': 0.10,
            'availability': 0.05,
            'mentor_type': 0.05,
            'experience_level': 0.05,
            'mentor_quality': 0.03,
            'performance_compatibility': 0.02
        }
        
        overall_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        return overall_score, factors
    
    def _calculate_major_compatibility(self, mentee_major: str, mentor_major: str, mentor_department: str) -> float:
        """Calculate compatibility based on academic majors and departments"""
        # Exact major match
        if mentee_major == mentor_major:
            return 1.0
        
        # Check if mentee's major contains mentor's department
        if mentor_department.lower() in mentee_major.lower():
            return 0.9
        
        # Cross-disciplinary compatibility
        compatible_pairs = {
            'Computer Science': ['Mathematics', 'Engineering'],
            'Biology': ['Chemistry', 'Mathematics'],
            'Mathematics': ['Computer Science', 'Physics'],
            'Chemistry': ['Biology', 'Physics'],
            'Physics': ['Mathematics', 'Engineering']
        }
        
        mentee_field = mentee_major.split('-')[0] if '-' in mentee_major else mentee_major
        
        if mentor_department in compatible_pairs.get(mentee_field, []):
            return 0.7
        elif mentee_field in compatible_pairs.get(mentor_department, []):
            return 0.7
        
        return 0.4  # Low but not zero compatibility
    
    def _calculate_learning_compatibility(self, mentee_id: str, mentor_id: str) -> float:
        """Calculate learning compatibility using performance similarity data"""
        if self.performance_similarity_df is not None:
            # Check if there's a performance similarity relationship
            similarity = self.performance_similarity_df[
                ((self.performance_similarity_df[':START_ID(Student)'] == mentee_id) & 
                 (self.performance_similarity_df[':END_ID(Student)'] == mentor_id)) |
                ((self.performance_similarity_df[':START_ID(Student)'] == mentor_id) & 
                 (self.performance_similarity_df[':END_ID(Student)'] == mentee_id))
            ]
            if not similarity.empty:
                return float(similarity.iloc[0]['similarity:float'])
        
        # Check learning style similarity if available
        if hasattr(self, 'learning_similarity_df') and self.learning_similarity_df is not None:
            learning_sim = self.learning_similarity_df[
                ((self.learning_similarity_df[':START_ID(Student)'] == mentee_id) & 
                 (self.learning_similarity_df[':END_ID(Student)'] == mentor_id)) |
                ((self.learning_similarity_df[':START_ID(Student)'] == mentor_id) & 
                 (self.learning_similarity_df[':END_ID(Student)'] == mentee_id))
            ]
            if not learning_sim.empty:
                return float(learning_sim.iloc[0]['similarity:float'])
        
        return 0.5  # Neutral score if no data available
    
    def _calculate_performance_compatibility(self, mentee_id: str, mentor_id: str) -> float:
        """Calculate performance compatibility - mentors should be slightly better performers"""
        if self.completed_df is not None:
            # Get average grades for both
            mentee_grades = self._get_average_grade(mentee_id)
            mentor_grades = self._get_average_grade(mentor_id)
            
            if mentee_grades is not None and mentor_grades is not None:
                # Ideal if mentor has slightly better grades (0.2-0.8 point difference)
                grade_diff = mentor_grades - mentee_grades
                if 0.2 <= grade_diff <= 0.8:
                    return 1.0
                elif 0.0 <= grade_diff <= 1.2:
                    return 0.8
                elif grade_diff > 1.2:
                    return 0.6  # Mentor much better - might be intimidating
                else:
                    return 0.4  # Mentor worse performer
        
        return 0.6  # Default moderate score
    
    def _get_average_grade(self, student_id: str) -> float:
        """Get average grade for a student"""
        if self.completed_df is not None:
            student_courses = self.completed_df[self.completed_df[':START_ID(Student)'] == student_id]
            if not student_courses.empty:
                grade_mapping = {'A+': 4.0, 'A': 4.0, 'A-': 3.7, 'B+': 3.3, 'B': 3.0, 'B-': 2.7, 'C+': 2.3, 'C': 2.0, 'C-': 1.7, 'D': 1.0, 'F': 0.0}
                grades = student_courses['grade'].tolist()
                gpa_sum = sum(grade_mapping.get(grade, 2.5) for grade in grades)
                return gpa_sum / len(grades) if grades else None
        return None
    
    def _calculate_interest_overlap(self, mentee_interests: List[str], mentor_interests: List[str]) -> float:
        """Calculate overlap between interests"""
        if not mentee_interests or not mentor_interests:
            return 0.5
        
        mentee_set = set([interest.lower() for interest in mentee_interests])
        mentor_set = set([interest.lower() for interest in mentor_interests])
        
        overlap = len(mentee_set.intersection(mentor_set))
        total = len(mentee_set.union(mentor_set))
        
        return overlap / total if total > 0 else 0.5
    
    def _calculate_career_alignment(self, career_goals: List[str], industry_experience: List[str]) -> float:
        """Calculate how well mentor's experience aligns with mentee's career goals"""
        if not career_goals or not industry_experience:
            return 0.5
        
        # Simple keyword matching
        goal_keywords = set([goal.lower() for goal in career_goals])
        exp_keywords = set([exp.lower() for exp in industry_experience])
        
        overlap = len(goal_keywords.intersection(exp_keywords))
        return min(overlap / len(goal_keywords), 1.0) if goal_keywords else 0.5
    
    def _calculate_skills_overlap(self, skills_to_develop: List[str], mentor_skills: List[str]) -> float:
        """Calculate how well mentor can help develop desired skills"""
        if not skills_to_develop or not mentor_skills:
            return 0.5
        
        develop_set = set([skill.lower() for skill in skills_to_develop])
        mentor_set = set([skill.lower() for skill in mentor_skills])
        
        overlap = len(develop_set.intersection(mentor_set))
        return overlap / len(develop_set) if develop_set else 0.5
    
    def _calculate_availability_compatibility(self, mentee_avail: str, mentor_avail: str) -> float:
        """Calculate availability compatibility"""
        if mentee_avail.lower() == "flexible" or mentor_avail.lower() == "flexible":
            return 1.0
        if mentee_avail.lower() == mentor_avail.lower():
            return 1.0
        return 0.6  # Different but potentially workable
    
    def _calculate_experience_appropriateness(self, mentee_year: int, mentor_experience: int, mentor_type: str) -> float:
        """Calculate if mentor's experience level is appropriate for mentee"""
        if mentor_type == "senior_student":
            # Senior students should be at least 2 years ahead
            gap = mentor_experience - mentee_year
            return min(gap / 2.0, 1.0) if gap >= 1 else 0.3
        elif mentor_type == "faculty":
            # Faculty experience should be substantial
            return min(mentor_experience / 10.0, 1.0)
        else:  # industry_professional
            # Industry experience should be relevant
            return min(mentor_experience / 5.0, 1.0)
    
    def _suggest_meeting_frequency(self, mentor_type: str) -> str:
        """Suggest meeting frequency based on mentor type"""
        frequencies = {
            "senior_student": "Weekly (1 hour)",
            "faculty": "Bi-weekly (30-45 minutes)",
            "industry_professional": "Monthly (1 hour)"
        }
        return frequencies.get(mentor_type, "Bi-weekly")
    
    def _suggest_activities(self, mentee: MenteeProfile, mentor: MentorProfile) -> List[str]:
        """Suggest mentorship activities based on profiles"""
        activities = []
        
        if mentor.mentor_type == "senior_student":
            activities.extend([
                "Study session planning",
                "Course selection guidance",
                "Time management coaching",
                "Peer network introduction"
            ])
        
        if mentor.mentor_type == "faculty":
            activities.extend([
                "Research opportunity discussions",
                "Graduate school preparation",
                "Academic paper reviews",
                "Conference attendance planning"
            ])
        
        if mentor.mentor_type == "industry_professional":
            activities.extend([
                "Resume and interview preparation",
                "Industry insights sharing",
                "Networking opportunities",
                "Internship guidance"
            ])
        
        # Add common activities
        activities.extend([
            "Goal setting and progress tracking",
            "Skill development planning",
            "Career path exploration"
        ])
        
        return activities[:6]  # Return top 6 activities
    
    def _suggest_goals(self, mentee: MenteeProfile, mentor: MentorProfile) -> List[str]:
        """Suggest mentorship goals"""
        goals = []
        
        if mentee.year_level <= 2:
            goals.extend([
                "Develop strong academic foundation",
                "Explore career options",
                "Build professional network"
            ])
        else:
            goals.extend([
                "Prepare for career transition",
                "Develop specialized skills",
                "Create professional portfolio"
            ])
        
        if mentor.mentor_type == "industry_professional":
            goals.append("Gain industry insights and connections")
        
        if mentor.mentor_type == "faculty":
            goals.append("Explore research and graduate opportunities")
        
        return goals[:4]
    
    def get_mentor_statistics(self) -> Dict[str, Any]:
        """Get mentorship program statistics"""
        
        total_mentors = len(self.senior_mentors) + len(self.faculty_mentors)
        
        return {
            "total_mentors": total_mentors,
            "senior_student_mentors": len(self.senior_mentors),
            "faculty_mentors": len(self.faculty_mentors),
            "average_rating": round(np.mean([m.rating for m in 
                                           self.senior_mentors + self.faculty_mentors]), 2),
            "top_research_areas": self._get_top_research_areas(),
            "mentor_distribution": {
                "senior_students": len(self.senior_mentors),
                "faculty": len(self.faculty_mentors)
            }
        }
    
    def _get_top_research_areas(self) -> List[Dict[str, Any]]:
        """Get most common research areas across all mentors"""
        all_areas = []
        for mentor in self.senior_mentors + self.faculty_mentors:
            all_areas.extend(mentor.research_interests)
        
        area_counts = defaultdict(int)
        for area in all_areas:
            area_counts[area] += 1
        
        return [{"area": area, "count": count} for area, count in 
                sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    def get_student_details(self, student_id: str) -> Dict[str, Any]:
        """Get student details by ID"""
        try:
            if self.students_df is None:
                logger.error("Students dataframe not loaded")
                return None
                
            logger.info(f"Looking for student {student_id} in dataframe with {len(self.students_df)} students")
            logger.info(f"Students dataframe columns: {self.students_df.columns.tolist()}")
            
            # Find student in dataframe
            student_info = self.students_df[self.students_df['id:ID(Student)'] == student_id]
            if student_info.empty:
                logger.warning(f"Student {student_id} not found")
                # Debug: show first few student IDs
                logger.info(f"Sample student IDs: {self.students_df['id:ID(Student)'].head().tolist()}")
                return None
                
            student = student_info.iloc[0]
            
            # Get student's major from degree relationships
            major = self._get_student_major(student_id, self.degree_df)
            
            # Calculate year level based on enrollment date and completed courses
            year_level = self._calculate_year_level(student_id)
            
            return {
                "student_id": student_id,
                "name": student.get('name', 'Unknown'),
                "major": major,
                "year_level": year_level,
                "learning_style": student.get('learningStyle', 'Unknown'),
                "preferred_pace": student.get('preferredPace', 'Unknown'),
                "work_hours": student.get('workHoursPerWeek:int', 0),
                "instruction_mode": student.get('preferredInstructionMode', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting student details for {student_id}: {e}")
            return None
    
    def create_mentorship_program(self, program_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new mentorship program"""
        
        program = {
            "program_id": f"PROG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": program_data.get("name", "New Mentorship Program"),
            "description": program_data.get("description", ""),
            "duration": program_data.get("duration", "1 semester"),
            "target_audience": program_data.get("target_audience", "All students"),
            "mentor_requirements": program_data.get("mentor_requirements", []),
            "activities": program_data.get("activities", []),
            "goals": program_data.get("goals", []),
            "created_date": datetime.now().isoformat(),
            "status": "active"
        }
        
        return program

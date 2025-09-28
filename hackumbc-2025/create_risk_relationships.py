#!/usr/bin/env python3
"""
Risk Relationship Creator for Student-Course Risk Prediction

This script creates risk relationships between students and courses based on multiple factors:
1. Similar student performance patterns
2. Course difficulty and workload compatibility
3. Learning style matching
4. Historical performance in similar courses
5. Prerequisites and background compatibility
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RiskRelationshipCreator:
    def __init__(self, data_path="ml/notebooks/umbc_data/csv/"):
        self.data_path = data_path
        self.students_df = None
        self.courses_df = None
        self.completed_courses_df = None
        self.enrolled_courses_df = None
        self.learning_similarity_df = None
        self.performance_similarity_df = None
        self.course_difficulty_similarity_df = None
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")
        
        # Load core datasets
        self.students_df = pd.read_csv(f"{self.data_path}students.csv")
        self.courses_df = pd.read_csv(f"{self.data_path}courses.csv")
        self.completed_courses_df = pd.read_csv(f"{self.data_path}completed_courses.csv")
        self.enrolled_courses_df = pd.read_csv(f"{self.data_path}enrolled_courses.csv")
        
        # Load similarity datasets
        self.learning_similarity_df = pd.read_csv(f"{self.data_path}learning_style_similarity.csv")
        self.performance_similarity_df = pd.read_csv(f"{self.data_path}performance_similarity.csv")
        self.course_difficulty_similarity_df = pd.read_csv(f"{self.data_path}course_similarity_difficulty.csv")
        
        print(f"Loaded {len(self.students_df)} students, {len(self.courses_df)} courses")
        print(f"Loaded {len(self.completed_courses_df)} completed course records")
        
    def calculate_grade_numeric(self, grade):
        """Convert letter grades to numeric values"""
        grade_map = {
            'A+': 4.0, 'A': 4.0, 'A-': 3.7,
            'B+': 3.3, 'B': 3.0, 'B-': 2.7,
            'C+': 2.3, 'C': 2.0, 'C-': 1.7,
            'D+': 1.3, 'D': 1.0, 'D-': 0.7,
            'F': 0.0
        }
        return grade_map.get(grade, 2.0)  # Default to C if grade not found
    
    def get_student_gpa(self, student_id):
        """Calculate student's GPA from completed courses"""
        student_courses = self.completed_courses_df[
            self.completed_courses_df[':START_ID(Student)'] == student_id
        ]
        
        if len(student_courses) == 0:
            return 2.5  # Default GPA for new students
        
        grades = [self.calculate_grade_numeric(grade) for grade in student_courses['grade']]
        return np.mean(grades)
    
    def get_similar_students(self, student_id, similarity_type='performance', top_k=10):
        """Get similar students based on learning style or performance"""
        if similarity_type == 'performance':
            similarity_df = self.performance_similarity_df
        else:
            similarity_df = self.learning_similarity_df
        
        # Get students similar to the target student
        similar_students = similarity_df[
            similarity_df[':START_ID(Student)'] == student_id
        ].nlargest(top_k, 'similarity:float')
        
        return similar_students[':END_ID(Student)'].tolist()
    
    def get_course_difficulty_score(self, course_id):
        """Get course difficulty score"""
        course = self.courses_df[self.courses_df['id:ID(Course)'] == course_id]
        if len(course) == 0:
            return 3.0  # Default difficulty
        
        return course['avgDifficulty:float'].iloc[0]
    
    def get_course_time_commitment(self, course_id):
        """Get course time commitment"""
        course = self.courses_df[self.courses_df['id:ID(Course)'] == course_id]
        if len(course) == 0:
            return 6  # Default time commitment
        
        return course['avgTimeCommitment:int'].iloc[0]
    
    def get_learning_style_match(self, student_id, course_id):
        """Calculate learning style match between student and course"""
        student = self.students_df[self.students_df['id:ID(Student)'] == student_id]
        course = self.courses_df[self.courses_df['id:ID(Course)'] == course_id]
        
        if len(student) == 0 or len(course) == 0:
            return 0.5  # Default match
        
        learning_style = student['learningStyle'].iloc[0]
        
        # Map learning style to course success rates
        style_mapping = {
            'Visual': 'visualLearnerSuccess:float',
            'Auditory': 'auditoryLearnerSuccess:float',
            'Kinesthetic': 'kinestheticLearnerSuccess:float',
            'Reading-Writing': 'readingLearnerSuccess:float'
        }
        
        success_column = style_mapping.get(learning_style, 'visualLearnerSuccess:float')
        return course[success_column].iloc[0]
    
    def get_workload_compatibility(self, student_id, course_id):
        """Calculate workload compatibility"""
        student = self.students_df[self.students_df['id:ID(Student)'] == student_id]
        course = self.courses_df[self.courses_df['id:ID(Course)'] == course_id]
        
        if len(student) == 0 or len(course) == 0:
            return 0.5
        
        student_preferred_load = student['preferredCourseLoad:int'].iloc[0]
        course_time_commitment = course['avgTimeCommitment:int'].iloc[0]
        student_work_hours = student['workHoursPerWeek:int'].iloc[0]
        
        # Calculate available study time (assuming 40 hours total per week)
        available_study_time = 40 - student_work_hours
        
        # Normalize course time commitment to 0-1 scale
        normalized_course_time = min(course_time_commitment / 15, 1.0)  # Max 15 hours per week
        
        # Calculate compatibility based on available time vs course requirements
        if available_study_time >= course_time_commitment:
            workload_score = 1.0
        else:
            workload_score = available_study_time / course_time_commitment
        
        return workload_score
    
    def get_historical_performance_in_similar_courses(self, student_id, course_id):
        """Get historical performance in courses similar to the target course"""
        # Get similar courses based on difficulty
        similar_courses = self.course_difficulty_similarity_df[
            self.course_difficulty_similarity_df[':START_ID(Course)'] == course_id
        ]
        
        if len(similar_courses) == 0:
            return 2.5  # Default performance
        
        similar_course_ids = similar_courses[':END_ID(Course)'].tolist()
        
        # Get student's performance in these similar courses
        student_performance = self.completed_courses_df[
            (self.completed_courses_df[':START_ID(Student)'] == student_id) &
            (self.completed_courses_df[':END_ID(Course)'].isin(similar_course_ids))
        ]
        
        if len(student_performance) == 0:
            return 2.5  # Default if no similar courses taken
        
        grades = [self.calculate_grade_numeric(grade) for grade in student_performance['grade']]
        return np.mean(grades)
    
    def get_similar_students_performance(self, student_id, course_id):
        """Get performance of similar students in the target course"""
        similar_students = self.get_similar_students(student_id, 'performance', top_k=5)
        
        if not similar_students:
            return 2.5  # Default performance
        
        # Get performance of similar students in this course
        similar_students_performance = self.completed_courses_df[
            (self.completed_courses_df[':START_ID(Student)'].isin(similar_students)) &
            (self.completed_courses_df[':END_ID(Course)'] == course_id)
        ]
        
        if len(similar_students_performance) == 0:
            return 2.5  # Default if no similar students took this course
        
        grades = [self.calculate_grade_numeric(grade) for grade in similar_students_performance['grade']]
        return np.mean(grades)
    
    def calculate_risk_score(self, student_id, course_id):
        """
        Calculate comprehensive risk score for student-course pair
        
        Risk factors (lower values = higher risk):
        1. Student GPA (0-4 scale)
        2. Course difficulty (1-5 scale, inverted)
        3. Learning style match (0-1 scale, inverted)
        4. Workload compatibility (0-1 scale, inverted)
        5. Historical performance in similar courses (0-4 scale, inverted)
        6. Similar students' performance (0-4 scale, inverted)
        """
        
        # Get individual risk factors
        student_gpa = self.get_student_gpa(student_id)
        course_difficulty = self.get_course_difficulty_score(course_id)
        learning_style_match = self.get_learning_style_match(student_id, course_id)
        workload_compatibility = self.get_workload_compatibility(student_id, course_id)
        historical_performance = self.get_historical_performance_in_similar_courses(student_id, course_id)
        similar_students_performance = self.get_similar_students_performance(student_id, course_id)
        
        # Normalize factors to 0-1 scale (higher = lower risk)
        gpa_factor = student_gpa / 4.0
        difficulty_factor = 1.0 - (course_difficulty - 1) / 4.0  # Invert difficulty
        learning_factor = learning_style_match
        workload_factor = workload_compatibility
        historical_factor = historical_performance / 4.0
        similar_students_factor = similar_students_performance / 4.0
        
        # Weighted combination of factors
        weights = {
            'gpa': 0.25,
            'difficulty': 0.20,
            'learning_style': 0.15,
            'workload': 0.15,
            'historical': 0.15,
            'similar_students': 0.10
        }
        
        risk_score = (
            weights['gpa'] * gpa_factor +
            weights['difficulty'] * difficulty_factor +
            weights['learning_style'] * learning_factor +
            weights['workload'] * workload_factor +
            weights['historical'] * historical_factor +
            weights['similar_students'] * similar_students_factor
        )
        
        # Convert to risk level (0-1 scale where 1 = highest risk)
        risk_level = 1.0 - risk_score
        
        return {
            'risk_score': risk_level,
            'risk_level': self.categorize_risk(risk_level),
            'pass_probability': risk_score,
            'factors': {
                'student_gpa': student_gpa,
                'course_difficulty': course_difficulty,
                'learning_style_match': learning_style_match,
                'workload_compatibility': workload_compatibility,
                'historical_performance': historical_performance,
                'similar_students_performance': similar_students_performance
            }
        }
    
    def categorize_risk(self, risk_score):
        """Categorize risk score into risk levels"""
        if risk_score < 0.2:
            return "LOW"
        elif risk_score < 0.4:
            return "MEDIUM"
        elif risk_score < 0.6:
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def create_risk_relationships(self, output_file="risk_relationships.csv"):
        """Create risk relationships for all student-course combinations"""
        print("Creating risk relationships...")
        
        risk_relationships = []
        
        # Get all unique student-course pairs from enrolled courses
        enrolled_pairs = self.enrolled_courses_df[
            [':START_ID(Student)', ':END_ID(Course)']
        ].drop_duplicates()
        
        total_pairs = len(enrolled_pairs)
        print(f"Processing {total_pairs} student-course pairs...")
        
        for idx, (_, row) in enumerate(enrolled_pairs.iterrows()):
            student_id = row[':START_ID(Student)']
            course_id = row[':END_ID(Course)']
            
            if idx % 100 == 0:
                print(f"Processed {idx}/{total_pairs} pairs...")
            
            try:
                risk_data = self.calculate_risk_score(student_id, course_id)
                
                risk_relationships.append({
                    ':START_ID(Student)': student_id,
                    ':END_ID(Course)': course_id,
                    ':TYPE': 'RISK',
                    'risk_score:float': risk_data['risk_score'],
                    'risk_level:string': risk_data['risk_level'],
                    'pass_probability:float': risk_data['pass_probability'],
                    'student_gpa:float': risk_data['factors']['student_gpa'],
                    'course_difficulty:float': risk_data['factors']['course_difficulty'],
                    'learning_style_match:float': risk_data['factors']['learning_style_match'],
                    'workload_compatibility:float': risk_data['factors']['workload_compatibility'],
                    'historical_performance:float': risk_data['factors']['historical_performance'],
                    'similar_students_performance:float': risk_data['factors']['similar_students_performance']
                })
                
            except Exception as e:
                print(f"Error processing {student_id}-{course_id}: {e}")
                continue
        
        # Create DataFrame and save
        risk_df = pd.DataFrame(risk_relationships)
        risk_df.to_csv(output_file, index=False)
        
        print(f"Created {len(risk_df)} risk relationships")
        print(f"Risk distribution:")
        print(risk_df['risk_level:string'].value_counts())
        
        return risk_df
    
    def generate_risk_summary(self, risk_df):
        """Generate summary statistics for risk relationships"""
        print("\n=== RISK RELATIONSHIP SUMMARY ===")
        
        # Overall statistics
        print(f"Total risk relationships: {len(risk_df)}")
        print(f"Average risk score: {risk_df['risk_score:float'].mean():.3f}")
        print(f"Average pass probability: {risk_df['pass_probability:float'].mean():.3f}")
        
        # Risk level distribution
        print("\nRisk Level Distribution:")
        risk_dist = risk_df['risk_level:string'].value_counts()
        for level, count in risk_dist.items():
            percentage = (count / len(risk_df)) * 100
            print(f"  {level}: {count} ({percentage:.1f}%)")
        
        # Top riskiest courses
        print("\nTop 10 Riskiest Courses (by average risk score):")
        course_risk = risk_df.groupby(':END_ID(Course)')['risk_score:float'].mean().sort_values(ascending=False)
        for course, risk in course_risk.head(10).items():
            print(f"  {course}: {risk:.3f}")
        
        # Students with highest risk
        print("\nTop 10 Students with Highest Average Risk:")
        student_risk = risk_df.groupby(':START_ID(Student)')['risk_score:float'].mean().sort_values(ascending=False)
        for student, risk in student_risk.head(10).items():
            print(f"  {student}: {risk:.3f}")

def main():
    """Main function to create risk relationships"""
    print("=== Student-Course Risk Relationship Creator ===")
    
    # Initialize creator
    creator = RiskRelationshipCreator()
    
    # Load data
    creator.load_data()
    
    # Create risk relationships
    risk_df = creator.create_risk_relationships("risk_relationships.csv")
    
    # Generate summary
    creator.generate_risk_summary(risk_df)
    
    print("\nRisk relationships saved to 'risk_relationships.csv'")
    print("This file can be imported into Neo4j to create RISK relationships between students and courses.")

if __name__ == "__main__":
    main()

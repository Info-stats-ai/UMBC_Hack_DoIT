"""
Study Groups Service - Find compatible study partners using multiple factors
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
import logging
from dataclasses import dataclass
from collections import defaultdict
import os

logger = logging.getLogger(__name__)

@dataclass
class StudyPartner:
    """Represents a potential study partner"""
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

@dataclass
class StudyGroup:
    """Represents a study group recommendation"""
    group_id: str
    course_id: str
    course_name: str
    members: List[StudyPartner]
    avg_compatibility: float
    recommended_meeting_time: str
    group_size: int
    learning_style_diversity: float
    performance_balance: float

class StudyGroupsService:
    """Service for finding compatible study partners and forming study groups"""
    
    def __init__(self, neo4j_driver=None, csv_data_path="../../ml/notebooks/umbc_data/csv"):
        self.driver = neo4j_driver
        self.csv_path = csv_data_path
        self.students_df = None
        self.courses_df = None
        self.enrolled_df = None
        self.completed_df = None
        self.learning_similarity_df = None
        self.performance_similarity_df = None
        
        # Load CSV data
        self._load_csv_data()
    
    def _load_csv_data(self):
        """Load CSV data for analysis"""
        try:
            base_path = os.path.join(os.path.dirname(__file__), self.csv_path)
            logger.info(f"Loading CSV data from: {base_path}")
            
            # Load student data
            students_path = os.path.join(base_path, "students.csv")
            if os.path.exists(students_path):
                self.students_df = pd.read_csv(students_path)
                logger.info(f"Loaded {len(self.students_df)} students")
                logger.info(f"Student DataFrame columns: {list(self.students_df.columns)}")
            
            # Load courses data
            courses_path = os.path.join(base_path, "courses.csv")
            if os.path.exists(courses_path):
                self.courses_df = pd.read_csv(courses_path)
                logger.info(f"Loaded {len(self.courses_df)} courses")
            
            # Load enrolled courses
            enrolled_path = os.path.join(base_path, "enrolled_courses.csv")
            if os.path.exists(enrolled_path):
                self.enrolled_df = pd.read_csv(enrolled_path)
                logger.info(f"Loaded {len(self.enrolled_df)} enrolled course records")
            
            # Load completed courses
            completed_path = os.path.join(base_path, "completed_courses.csv")
            if os.path.exists(completed_path):
                self.completed_df = pd.read_csv(completed_path)
                logger.info(f"Loaded {len(self.completed_df)} completed course records")
            
            # Load learning style similarity
            learning_sim_path = os.path.join(base_path, "learning_style_similarity.csv")
            if os.path.exists(learning_sim_path):
                self.learning_similarity_df = pd.read_csv(learning_sim_path)
                logger.info(f"Loaded {len(self.learning_similarity_df)} learning style similarity records")
            
            # Load performance similarity
            performance_sim_path = os.path.join(base_path, "performance_similarity.csv")
            if os.path.exists(performance_sim_path):
                self.performance_similarity_df = pd.read_csv(performance_sim_path)
                logger.info(f"Loaded {len(self.performance_similarity_df)} performance similarity records")
                
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
    
    def find_study_partners(self, student_id: str, course_id: str, max_partners: int = 10) -> List[StudyPartner]:
        """Find compatible study partners for a student in a specific course"""
        
        logger.info(f"Finding study partners for student {student_id} in course {course_id}")
        
        if not self._data_loaded():
            logger.warning("CSV data not loaded properly")
            return []
        
        try:
            # Get the target student's information
            target_student = self._get_student_info(student_id)
            if not target_student:
                logger.warning(f"Target student {student_id} not found")
                return []
            
            logger.info(f"Found target student: {target_student}")
            
            # Find other students enrolled in the same course
            course_students = self._get_students_in_course(course_id, exclude_student=student_id)
            logger.info(f"Found {len(course_students)} other students in course {course_id}")
            
            if not course_students:
                logger.warning(f"No other students found in course {course_id}")
                return []
            
            # Calculate compatibility scores for each potential partner
            partners = []
            for partner_id in course_students:
                logger.info(f"Processing potential partner: {partner_id}")
                partner_info = self._get_student_info(partner_id)
                if partner_info:
                    logger.info(f"Partner info: {partner_info}")
                    try:
                        compatibility_score, factors = self._calculate_compatibility(
                            target_student, partner_info, course_id
                        )
                    except Exception as e:
                        logger.error(f"Error calculating compatibility with {partner_id}: {e}")
                        continue
                    
                    if compatibility_score > 0.3:  # Minimum compatibility threshold
                        partner = StudyPartner(
                            student_id=partner_id,
                            name=partner_info['name'],
                            learning_style=partner_info['learningStyle'],
                            current_courses=self._get_student_current_courses(partner_id),
                            completed_courses=self._get_student_completed_courses(partner_id),
                            performance_level=self._get_performance_level(partner_id),
                            preferred_pace=partner_info['preferredPace'],
                            work_hours=partner_info['workHoursPerWeek:int'],
                            instruction_mode=partner_info['preferredInstructionMode'],
                            compatibility_score=compatibility_score,
                            compatibility_factors=factors
                        )
                        partners.append(partner)
            
            # Sort by compatibility score and return top matches
            partners.sort(key=lambda x: x.compatibility_score, reverse=True)
            return partners[:max_partners]
            
        except Exception as e:
            logger.error(f"Error finding study partners: {e}")
            return []
    
    def create_study_groups(self, course_id: str, min_group_size: int = 3, max_group_size: int = 5) -> List[StudyGroup]:
        """Create optimal study groups for a course using clustering algorithms"""
        
        if not self._data_loaded():
            return []
        
        try:
            # Get all students enrolled in the course
            course_students = self._get_students_in_course(course_id)
            
            if len(course_students) < min_group_size:
                return []
            
            # Calculate compatibility matrix
            compatibility_matrix = self._calculate_compatibility_matrix(course_students, course_id)
            
            # Use greedy clustering to form groups
            groups = self._form_groups_greedy(
                course_students, compatibility_matrix, min_group_size, max_group_size
            )
            
            # Convert to StudyGroup objects
            study_groups = []
            course_info = self._get_course_info(course_id)
            course_name = course_info['name'] if course_info else f"Course {course_id}"
            
            for i, group_members in enumerate(groups):
                if len(group_members) >= min_group_size:
                    members = []
                    for student_id in group_members:
                        student_info = self._get_student_info(student_id)
                        if student_info:
                            # Calculate average compatibility with other group members
                            avg_compatibility = self._calculate_group_compatibility(
                                student_id, group_members, compatibility_matrix
                            )
                            
                            partner = StudyPartner(
                                student_id=student_id,
                                name=student_info['name'],
                                learning_style=student_info['learningStyle'],
                                current_courses=self._get_student_current_courses(student_id),
                                completed_courses=self._get_student_completed_courses(student_id),
                                performance_level=self._get_performance_level(student_id),
                                preferred_pace=student_info['preferredPace'],
                                work_hours=student_info['workHoursPerWeek:int'],
                                instruction_mode=student_info['preferredInstructionMode'],
                                compatibility_score=avg_compatibility,
                                compatibility_factors={}
                            )
                            members.append(partner)
                    
                    if members:
                        group = StudyGroup(
                            group_id=f"GROUP_{course_id}_{i+1}",
                            course_id=course_id,
                            course_name=course_name,
                            members=members,
                            avg_compatibility=np.mean([m.compatibility_score for m in members]),
                            recommended_meeting_time=self._suggest_meeting_time(members),
                            group_size=len(members),
                            learning_style_diversity=self._calculate_learning_diversity(members),
                            performance_balance=self._calculate_performance_balance(members)
                        )
                        study_groups.append(group)
            
            return study_groups
            
        except Exception as e:
            logger.error(f"Error creating study groups: {e}")
            return []
    
    def _data_loaded(self) -> bool:
        """Check if required data is loaded"""
        return (self.students_df is not None and 
                self.enrolled_df is not None and 
                self.courses_df is not None)
    
    def _get_student_info(self, student_id: str) -> Optional[Dict]:
        """Get student information"""
        try:
            student_row = self.students_df[self.students_df['id:ID(Student)'] == student_id]
            if not student_row.empty:
                return student_row.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting student info for {student_id}: {e}")
            return None
    
    def _get_course_info(self, course_id: str) -> Optional[Dict]:
        """Get course information"""
        try:
            course_row = self.courses_df[self.courses_df['id:ID(Course)'] == course_id]
            if not course_row.empty:
                return course_row.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting course info for {course_id}: {e}")
            return None
    
    def _get_students_in_course(self, course_id: str, exclude_student: str = None) -> List[str]:
        """Get list of students enrolled in a course"""
        try:
            enrolled_students = self.enrolled_df[
                self.enrolled_df[':END_ID(Course)'] == course_id
            ][':START_ID(Student)'].tolist()
            
            if exclude_student:
                enrolled_students = [s for s in enrolled_students if s != exclude_student]
            
            return enrolled_students
        except Exception as e:
            logger.error(f"Error getting students in course {course_id}: {e}")
            return []
    
    def _get_student_current_courses(self, student_id: str) -> List[str]:
        """Get current courses for a student"""
        try:
            courses = self.enrolled_df[
                self.enrolled_df[':START_ID(Student)'] == student_id
            ][':END_ID(Course)'].tolist()
            return courses
        except Exception as e:
            logger.error(f"Error getting current courses for {student_id}: {e}")
            return []
    
    def _get_student_completed_courses(self, student_id: str) -> List[str]:
        """Get completed courses for a student"""
        try:
            if self.completed_df is not None:
                courses = self.completed_df[
                    self.completed_df[':START_ID(Student)'] == student_id
                ][':END_ID(Course)'].tolist()
                return courses
            return []
        except Exception as e:
            logger.error(f"Error getting completed courses for {student_id}: {e}")
            return []
    
    def _get_performance_level(self, student_id: str) -> str:
        """Determine student's performance level based on completed courses"""
        try:
            if self.completed_df is not None:
                student_grades = self.completed_df[
                    self.completed_df[':START_ID(Student)'] == student_id
                ]
                
                if not student_grades.empty:
                    # Convert grades to numeric scores
                    grade_mapping = {
                        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
                        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
                        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
                        'F': 0.0
                    }
                    
                    grades = student_grades['grade'].map(grade_mapping).dropna()
                    if len(grades) > 0:
                        avg_gpa = grades.mean()
                        if avg_gpa >= 3.5:
                            return "High"
                        elif avg_gpa >= 2.5:
                            return "Medium"
                        else:
                            return "Low"
            
            return "Unknown"
        except Exception as e:
            logger.error(f"Error getting performance level for {student_id}: {e}")
            return "Unknown"
    
    def _calculate_compatibility(self, student1: Dict, student2: Dict, course_id: str) -> Tuple[float, Dict[str, float]]:
        """Calculate compatibility score between two students"""
        factors = {}
        
        # Learning style compatibility
        learning_style_score = self._get_learning_style_compatibility(
            student1['id:ID(Student)'], student2['id:ID(Student)']
        )
        factors['learning_style'] = learning_style_score
        
        # Performance similarity
        performance_score = self._get_performance_similarity(
            student1['id:ID(Student)'], student2['id:ID(Student)']
        )
        factors['performance'] = performance_score
        
        # Schedule compatibility
        schedule_score = self._calculate_schedule_compatibility(student1, student2)
        factors['schedule'] = schedule_score
        
        # Learning pace compatibility
        pace_score = 1.0 if student1['preferredPace'] == student2['preferredPace'] else 0.5
        factors['pace'] = pace_score
        
        # Instruction mode compatibility
        mode_score = self._calculate_instruction_mode_compatibility(student1, student2)
        factors['instruction_mode'] = mode_score
        
        # Course-specific compatibility
        course_score = self._calculate_course_compatibility(
            student1['id:ID(Student)'], student2['id:ID(Student)'], course_id
        )
        factors['course_specific'] = course_score
        
        # Weighted overall score
        weights = {
            'learning_style': 0.25,
            'performance': 0.20,
            'schedule': 0.20,
            'pace': 0.15,
            'instruction_mode': 0.10,
            'course_specific': 0.10
        }
        
        overall_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        return overall_score, factors
    
    def _get_learning_style_compatibility(self, student1_id: str, student2_id: str) -> float:
        """Get learning style compatibility from similarity data"""
        try:
            if self.learning_similarity_df is not None:
                # Check both directions
                similarity = self.learning_similarity_df[
                    ((self.learning_similarity_df[':START_ID(Student)'] == student1_id) &
                     (self.learning_similarity_df[':END_ID(Student)'] == student2_id)) |
                    ((self.learning_similarity_df[':START_ID(Student)'] == student2_id) &
                     (self.learning_similarity_df[':END_ID(Student)'] == student1_id))
                ]
                
                if not similarity.empty:
                    return float(similarity.iloc[0]['similarity:float'])
            
            # Fallback: basic learning style match
            student1_info = self._get_student_info(student1_id)
            student2_info = self._get_student_info(student2_id)
            
            if student1_info and student2_info:
                if student1_info['learningStyle'] == student2_info['learningStyle']:
                    return 0.8
                else:
                    return 0.4
            
            return 0.5
        except Exception as e:
            logger.error(f"Error getting learning style compatibility: {e}")
            return 0.5
    
    def _get_performance_similarity(self, student1_id: str, student2_id: str) -> float:
        """Get performance similarity from similarity data"""
        try:
            if self.performance_similarity_df is not None:
                # Check both directions
                similarity = self.performance_similarity_df[
                    ((self.performance_similarity_df[':START_ID(Student)'] == student1_id) &
                     (self.performance_similarity_df[':END_ID(Student)'] == student2_id)) |
                    ((self.performance_similarity_df[':START_ID(Student)'] == student2_id) &
                     (self.performance_similarity_df[':END_ID(Student)'] == student1_id))
                ]
                
                if not similarity.empty:
                    return float(similarity.iloc[0]['similarity:float'])
            
            return 0.5
        except Exception as e:
            logger.error(f"Error getting performance similarity: {e}")
            return 0.5
    
    def _calculate_schedule_compatibility(self, student1: Dict, student2: Dict) -> float:
        """Calculate schedule compatibility based on work hours"""
        try:
            work1 = student1['workHoursPerWeek:int']
            work2 = student2['workHoursPerWeek:int']
            
            # Students with similar work hours are more compatible
            diff = abs(work1 - work2)
            if diff <= 5:
                return 1.0
            elif diff <= 10:
                return 0.7
            elif diff <= 20:
                return 0.5
            else:
                return 0.3
        except Exception as e:
            logger.error(f"Error calculating schedule compatibility: {e}")
            return 0.5
    
    def _calculate_instruction_mode_compatibility(self, student1: Dict, student2: Dict) -> float:
        """Calculate instruction mode compatibility"""
        try:
            mode1 = student1['preferredInstructionMode']
            mode2 = student2['preferredInstructionMode']
            
            if mode1 == mode2:
                return 1.0
            elif mode1 == 'Hybrid' or mode2 == 'Hybrid':
                return 0.8  # Hybrid is compatible with both
            else:
                return 0.4  # Online vs In-person
        except Exception as e:
            logger.error(f"Error calculating instruction mode compatibility: {e}")
            return 0.5
    
    def _calculate_course_compatibility(self, student1_id: str, student2_id: str, course_id: str) -> float:
        """Calculate course-specific compatibility"""
        try:
            # Check if students have taken similar courses
            student1_courses = set(self._get_student_completed_courses(student1_id))
            student2_courses = set(self._get_student_completed_courses(student2_id))
            
            if student1_courses and student2_courses:
                overlap = len(student1_courses.intersection(student2_courses))
                total = len(student1_courses.union(student2_courses))
                return overlap / total if total > 0 else 0.5
            
            return 0.5
        except Exception as e:
            logger.error(f"Error calculating course compatibility: {e}")
            return 0.5
    
    def _calculate_compatibility_matrix(self, students: List[str], course_id: str) -> np.ndarray:
        """Calculate compatibility matrix for all students"""
        n = len(students)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                student1 = self._get_student_info(students[i])
                student2 = self._get_student_info(students[j])
                
                if student1 and student2:
                    score, _ = self._calculate_compatibility(student1, student2, course_id)
                    matrix[i][j] = score
                    matrix[j][i] = score  # Symmetric matrix
        
        return matrix
    
    def _form_groups_greedy(self, students: List[str], compatibility_matrix: np.ndarray, 
                           min_size: int, max_size: int) -> List[List[str]]:
        """Form groups using greedy algorithm"""
        groups = []
        remaining_students = list(range(len(students)))
        
        while len(remaining_students) >= min_size:
            # Start with the student who has highest average compatibility
            avg_compatibility = [
                np.mean([compatibility_matrix[i][j] for j in remaining_students if j != i])
                for i in remaining_students
            ]
            
            if not avg_compatibility:
                break
                
            best_student_idx = remaining_students[np.argmax(avg_compatibility)]
            current_group = [best_student_idx]
            remaining_students.remove(best_student_idx)
            
            # Add compatible students to the group
            while len(current_group) < max_size and remaining_students:
                # Find the most compatible remaining student
                best_score = -1
                best_candidate = None
                
                for candidate in remaining_students:
                    # Calculate average compatibility with current group
                    group_compatibility = np.mean([
                        compatibility_matrix[candidate][member] for member in current_group
                    ])
                    
                    if group_compatibility > best_score:
                        best_score = group_compatibility
                        best_candidate = candidate
                
                # Only add if compatibility is above threshold
                if best_candidate is not None and best_score > 0.4:
                    current_group.append(best_candidate)
                    remaining_students.remove(best_candidate)
                else:
                    break
            
            # Add group if it meets minimum size
            if len(current_group) >= min_size:
                groups.append([students[i] for i in current_group])
        
        return groups
    
    def _calculate_group_compatibility(self, student_id: str, group_members: List[str], 
                                     compatibility_matrix: np.ndarray) -> float:
        """Calculate average compatibility of a student with group members"""
        # This is a placeholder - would need the actual indices
        return 0.75  # Default value
    
    def _suggest_meeting_time(self, members: List[StudyPartner]) -> str:
        """Suggest optimal meeting time based on member schedules"""
        # Analyze work hours and suggest times
        work_hours = [member.work_hours for member in members]
        avg_work_hours = np.mean(work_hours)
        
        if avg_work_hours < 10:
            return "Mon/Wed 3-5 PM"
        elif avg_work_hours < 20:
            return "Tue/Thu 6-8 PM"
        else:
            return "Sat/Sun 2-4 PM"
    
    def _calculate_learning_diversity(self, members: List[StudyPartner]) -> float:
        """Calculate learning style diversity in the group"""
        learning_styles = [member.learning_style for member in members]
        unique_styles = len(set(learning_styles))
        return unique_styles / len(members) if members else 0
    
    def _calculate_performance_balance(self, members: List[StudyPartner]) -> float:
        """Calculate performance balance in the group"""
        performance_levels = [member.performance_level for member in members]
        level_counts = defaultdict(int)
        for level in performance_levels:
            level_counts[level] += 1
        
        # Balanced groups have members from different performance levels
        if len(level_counts) > 1:
            return 1.0 - max(level_counts.values()) / len(members)
        return 0.5

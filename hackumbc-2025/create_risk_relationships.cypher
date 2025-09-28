// Create Risk Relationships directly in Neo4j using existing data
// This query calculates risk scores based on student and course data

// First, let's create a function to calculate risk score
// We'll use the existing student and course data to compute risk

// Step 1: Create risk relationships for all enrolled student-course pairs
MATCH (s:Student)-[:ENROLLED_IN]->(c:Course)
WITH s, c

// Calculate student GPA from completed courses
OPTIONAL MATCH (s)-[:COMPLETED]->(completed:Course)
WITH s, c, 
     CASE 
       WHEN count(completed) > 0 
       THEN avg(
         CASE completed.grade
           WHEN 'A+' THEN 4.0
           WHEN 'A' THEN 4.0
           WHEN 'A-' THEN 3.7
           WHEN 'B+' THEN 3.3
           WHEN 'B' THEN 3.0
           WHEN 'B-' THEN 2.7
           WHEN 'C+' THEN 2.3
           WHEN 'C' THEN 2.0
           WHEN 'C-' THEN 1.7
           WHEN 'D+' THEN 1.3
           WHEN 'D' THEN 1.0
           WHEN 'D-' THEN 0.7
           WHEN 'F' THEN 0.0
           ELSE 2.0
         END
       )
       ELSE 2.5
     END AS student_gpa

// Get course difficulty and learning style match
WITH s, c, student_gpa,
     c.avgDifficulty AS course_difficulty,
     CASE s.learningStyle
       WHEN 'Visual' THEN c.visualLearnerSuccess
       WHEN 'Auditory' THEN c.auditoryLearnerSuccess
       WHEN 'Kinesthetic' THEN c.kinestheticLearnerSuccess
       WHEN 'Reading-Writing' THEN c.readingLearnerSuccess
       ELSE 0.5
     END AS learning_style_match

// Calculate workload compatibility
WITH s, c, student_gpa, course_difficulty, learning_style_match,
     CASE 
       WHEN (40 - s.workHoursPerWeek) >= c.avgTimeCommitment 
       THEN 1.0
       ELSE (40 - s.workHoursPerWeek) / c.avgTimeCommitment
     END AS workload_compatibility

// Get historical performance in similar courses
OPTIONAL MATCH (s)-[:COMPLETED]->(similar:Course)
WHERE similar.avgDifficulty = course_difficulty
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility,
     CASE 
       WHEN count(similar) > 0 
       THEN avg(
         CASE similar.grade
           WHEN 'A+' THEN 4.0
           WHEN 'A' THEN 4.0
           WHEN 'A-' THEN 3.7
           WHEN 'B+' THEN 3.3
           WHEN 'B' THEN 3.0
           WHEN 'B-' THEN 2.7
           WHEN 'C+' THEN 2.3
           WHEN 'C' THEN 2.0
           WHEN 'C-' THEN 1.7
           WHEN 'D+' THEN 1.3
           WHEN 'D' THEN 1.0
           WHEN 'D-' THEN 0.7
           WHEN 'F' THEN 0.0
           ELSE 2.0
         END
       )
       ELSE 2.5
     END AS historical_performance

// Get similar students' performance in this course
OPTIONAL MATCH (similar_student:Student)-[:COMPLETED]->(c)
WHERE similar_student.learningStyle = s.learningStyle
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility, historical_performance,
     CASE 
       WHEN count(similar_student) > 0 
       THEN avg(
         CASE c.grade
           WHEN 'A+' THEN 4.0
           WHEN 'A' THEN 4.0
           WHEN 'A-' THEN 3.7
           WHEN 'B+' THEN 3.3
           WHEN 'B' THEN 3.0
           WHEN 'B-' THEN 2.7
           WHEN 'C+' THEN 2.3
           WHEN 'C' THEN 2.0
           WHEN 'C-' THEN 1.7
           WHEN 'D+' THEN 1.3
           WHEN 'D' THEN 1.0
           WHEN 'D-' THEN 0.7
           WHEN 'F' THEN 0.0
           ELSE 2.0
         END
       )
       ELSE 2.5
     END AS similar_students_performance

// Calculate risk score
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility, historical_performance, similar_students_performance,
     // Normalize factors to 0-1 scale (higher = lower risk)
     (student_gpa / 4.0) AS gpa_factor,
     (1.0 - (course_difficulty - 1) / 4.0) AS difficulty_factor,
     learning_style_match AS learning_factor,
     workload_compatibility AS workload_factor,
     (historical_performance / 4.0) AS historical_factor,
     (similar_students_performance / 4.0) AS similar_students_factor

// Weighted combination of factors
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility, historical_performance, similar_students_performance,
     (0.25 * gpa_factor + 
      0.20 * difficulty_factor + 
      0.15 * learning_factor + 
      0.15 * workload_factor + 
      0.15 * historical_factor + 
      0.10 * similar_students_factor) AS pass_probability,
     (1.0 - (0.25 * gpa_factor + 
             0.20 * difficulty_factor + 
             0.15 * learning_factor + 
             0.15 * workload_factor + 
             0.15 * historical_factor + 
             0.10 * similar_students_factor)) AS risk_score

// Categorize risk level
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility, historical_performance, similar_students_performance, pass_probability, risk_score,
     CASE 
       WHEN risk_score < 0.2 THEN 'LOW'
       WHEN risk_score < 0.4 THEN 'MEDIUM'
       WHEN risk_score < 0.6 THEN 'HIGH'
       ELSE 'VERY_HIGH'
     END AS risk_level

// Create the RISK relationship
MERGE (s)-[r:RISK]->(c)
SET r.risk_score = risk_score,
    r.risk_level = risk_level,
    r.pass_probability = pass_probability,
    r.student_gpa = student_gpa,
    r.course_difficulty = course_difficulty,
    r.learning_style_match = learning_style_match,
    r.workload_compatibility = workload_compatibility,
    r.historical_performance = historical_performance,
    r.similar_students_performance = similar_students_performance;

// Verify the creation
MATCH (s:Student)-[r:RISK]->(c:Course)
RETURN count(r) as total_risk_relationships;

// Show sample risk relationships
MATCH (s:Student)-[r:RISK]->(c:Course)
RETURN s.id as student_id, c.id as course_id, r.risk_level, r.pass_probability, r.risk_score
ORDER BY r.risk_score DESC
LIMIT 10;

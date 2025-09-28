// Create Comprehensive Risk Relationships for ALL student-course combinations
// This allows risk prediction for any course, not just enrolled ones

// Step 1: Create risk relationships for ALL student-course pairs
MATCH (s:Student), (c:Course)
// Exclude courses already completed to avoid redundancy
WHERE NOT (s)-[:COMPLETED]->(c)
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
     COALESCE(c.avgDifficulty, 3.0) AS course_difficulty,
     CASE s.learningStyle
       WHEN 'Visual' THEN COALESCE(c.visualLearnerSuccess, 0.5)
       WHEN 'Auditory' THEN COALESCE(c.auditoryLearnerSuccess, 0.5)
       WHEN 'Kinesthetic' THEN COALESCE(c.kinestheticLearnerSuccess, 0.5)
       WHEN 'Reading-Writing' THEN COALESCE(c.readingLearnerSuccess, 0.5)
       ELSE 0.5
     END AS learning_style_match

// Calculate workload compatibility
WITH s, c, student_gpa, course_difficulty, learning_style_match,
     CASE
       WHEN (40 - COALESCE(s.workHoursPerWeek, 20)) >= COALESCE(c.avgTimeCommitment, 10)
       THEN 1.0
       ELSE (40 - COALESCE(s.workHoursPerWeek, 20)) / COALESCE(c.avgTimeCommitment, 10)
     END AS workload_compatibility

// Get historical performance in similar difficulty courses
OPTIONAL MATCH (s)-[:COMPLETED]->(similar:Course)
WHERE ABS(similar.avgDifficulty - course_difficulty) <= 0.5
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

// Calculate risk score for ALL courses
WITH s, c, student_gpa, course_difficulty, learning_style_match, workload_compatibility, historical_performance, similar_students_performance,
     // Normalize factors to 0-1 scale (higher = lower risk)
     (student_gpa / 4.0) AS gpa_factor,
     (1.0 - (course_difficulty - 1) / 4.0) AS difficulty_factor,
     learning_style_match AS learning_factor,
     CASE
       WHEN workload_compatibility > 1.0 THEN 1.0
       ELSE workload_compatibility
     END AS workload_factor,
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

// Create the RISK relationship for ALL student-course combinations
MERGE (s)-[r:RISK]->(c)
SET r.risk_score = risk_score,
    r.risk_level = risk_level,
    r.pass_probability = pass_probability,
    r.student_gpa = student_gpa,
    r.course_difficulty = course_difficulty,
    r.learning_style_match = learning_style_match,
    r.workload_compatibility = workload_compatibility,
    r.historical_performance = historical_performance,
    r.similar_students_performance = similar_students_performance,
    r.calculated_at = datetime();

// Verify the creation
MATCH (s:Student)-[r:RISK]->(c:Course)
RETURN count(r) as total_risk_relationships;

// Show sample risk relationships sorted by risk score
MATCH (s:Student {id: 'ZO28124'})-[r:RISK]->(c:Course)
RETURN s.id as student_id, c.id as course_id, c.name as course_name,
       r.risk_level, r.pass_probability, r.risk_score
ORDER BY r.risk_score DESC
LIMIT 10;
LOAD CSV WITH HEADERS FROM 'file:///risk_relationships.csv' AS row
MATCH (s:Student {id: row.`:START_ID(Student)`})
MATCH (c:Course  {id: row.`:END_ID(Course)`})
MERGE (s)-[r:RISK]->(c)
SET r.risk_score = toFloat(row.`risk_score:float`),
    r.risk_level = row.`risk_level:string`,
    r.pass_probability = toFloat(row.`pass_probability:float`),
    r.student_gpa = toFloat(row.`student_gpa:float`),
    r.course_difficulty = toFloat(row.`course_difficulty:float`),
    r.learning_style_match = toFloat(row.`learning_style_match:float`),
    r.workload_compatibility = toFloat(row.`workload_compatibility:float`),
    r.historical_performance = toFloat(row.`historical_performance:float`),
    r.similar_students_performance = toFloat(row.`similar_students_performance:float`);
# Load New Synthetic Dataset into Neo4j
import os
import glob

def clear_database(driver):
    """Clear all existing data from the database"""
    print("🧹 Clearing existing data...")
    with driver.session(database="neo4j") as session:
        # Clear all data
        session.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared successfully")

def load_cypher_files(driver, cypher_dir="umbc_data/cypher"):
    """Load all Cypher files in the correct order"""
    
    # Define the correct order for loading files
    file_order = [
        "00_indexes.cypher",
        "01_students.cypher", 
        "02_faculty.cypher",
        "03_terms.cypher",
        "04_courses.cypher", 
        "05_degrees.cypher",
        "06_requirement_groups.cypher",
        "07_course_prerequisites.cypher",
        "08_leads_to.cypher",
        "09_course_similarity.cypher",
        "10_student_degree.cypher",
        "11_teaching.cypher", 
        "12_completed_courses.cypher",
        "13_enrolled_courses.cypher",
        "14_student_similarity.cypher",
        "15_requirement_degree.cypher",
        "16_course_requirement.cypher",
        "17_course_term.cypher"
    ]
    
    print(f"📊 Loading {len(file_order)} Cypher files...")
    
    for i, filename in enumerate(file_order, 1):
        filepath = os.path.join(cypher_dir, filename)
        
        if os.path.exists(filepath):
            print(f"   [{i:2d}/{len(file_order)}] Loading {filename}...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    cypher_content = f.read()
                
                # Split by semicolon and execute each statement
                statements = [stmt.strip() for stmt in cypher_content.split(';') if stmt.strip()]
                
                with driver.session(database="neo4j") as session:
                    for stmt in statements:
                        if stmt:  # Only execute non-empty statements
                            session.run(stmt)
                
                print(f"   ✅ {filename} loaded successfully")
                
            except Exception as e:
                print(f"   ❌ Error loading {filename}: {str(e)}")
                print("   Continuing with next file...")
        else:
            print(f"   ⚠️  File not found: {filename}")

def verify_data_import(driver):
    """Verify that the data was imported correctly"""
    print("🔍 Verifying data import...")
    
    with driver.session(database="neo4j") as session:
        # Count nodes by type
        result = session.run("""
            MATCH (n) 
            RETURN labels(n)[0] as NodeType, count(n) as Count
            ORDER BY Count DESC
        """)
        
        print("\nNode counts after import:")
        print("-" * 30)
        total_nodes = 0
        for record in result:
            node_type = record["NodeType"]
            count = record["Count"]
            total_nodes += count
            print(f"{node_type:15} | {count:4d}")
        
        print(f"\nTotal nodes: {total_nodes}")
        
        # Count relationships
        rel_result = session.run("""
            MATCH ()-[r]->() 
            RETURN type(r) as RelType, count(r) as Count
            ORDER BY Count DESC
        """)
        
        print("\nRelationship counts:")
        print("-" * 30)
        total_rels = 0
        for record in rel_result:
            rel_type = record["RelType"]
            count = record["Count"]
            total_rels += count
            print(f"{rel_type:20} | {count:4d}")
        
        print(f"\nTotal relationships: {total_rels}")

# Execute the data loading process
if driver:
    print("🚀 Starting Neo4j Data Loading Process")
    print("=" * 40)
    
    # Step 1: Clear existing data
    clear_database(driver)
    
    # Step 2: Load new data
    load_cypher_files(driver)
    
    # Step 3: Verify import
    verify_data_import(driver)
    
    print("\n🎉 Data loading completed!")
    print("Your Neo4j database now contains the full synthetic dataset!")
    
else:
    print("❌ No database connection available")

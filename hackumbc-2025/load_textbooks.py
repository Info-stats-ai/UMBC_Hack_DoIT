from neo4j import GraphDatabase
import os

def load_textbooks():
    """Load textbooks from CSV into Neo4j database"""
    
    # Connect to Neo4j
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'umbctest123'))

    # Read the textbooks CSV file
    csv_path = 'ml/notebooks/umbc_data/csv/textbooks.csv'
    textbooks_data = []

    print('📖 Reading textbooks.csv...')
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # Skip header line
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 9:  # Ensure we have all required fields
                    textbooks_data.append({
                        'id': parts[0],
                        'name': parts[1],
                        'publisher': parts[2],
                        'price': float(parts[3]),
                        'pages': int(parts[4]),
                        'edition': int(parts[5]),
                        'publicationYear': int(parts[6]),
                        'isbn': parts[7],
                        'category': parts[8]
                    })

    print(f'📚 Found {len(textbooks_data)} textbooks to load')

    # Load textbooks into Neo4j
    with driver.session(database='neo4j') as session:
        print('🔄 Loading textbooks into Neo4j...')
        
        loaded_count = 0
        for i, textbook in enumerate(textbooks_data):
            try:
                session.run('''
                    CREATE (t:Textbook {
                        id: $id,
                        name: $name,
                        publisher: $publisher,
                        price: $price,
                        pages: $pages,
                        edition: $edition,
                        publicationYear: $publicationYear,
                        isbn: $isbn,
                        category: $category
                    })
                ''', **textbook)
                
                loaded_count += 1
                if (i + 1) % 20 == 0:
                    print(f'  Loaded {i + 1}/{len(textbooks_data)} textbooks...')
                    
            except Exception as e:
                print(f'❌ Error loading textbook {textbook["id"]}: {e}')
        
        print(f'✅ Successfully loaded {loaded_count} textbooks!')
        
        # Verify the load
        result = session.run('MATCH (t:Textbook) RETURN count(t) as count')
        count = result.single()['count']
        print(f'🔍 Verification: {count} textbooks now in database')

    driver.close()
    return loaded_count

if __name__ == "__main__":
    load_textbooks()

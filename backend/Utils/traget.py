import pandas as pd
import re

# Load your merged CSV
df = pd.read_csv('synthetic_queries_merged.csv')  # Replace with your actual path

def extract_index_columns(query):
    """
    Extracts candidate columns for indexing from SQL query.
    Looks into WHERE, JOIN, GROUP BY, ORDER BY clauses.
    Returns a comma-separated string of candidate columns.
    """
    query = query.lower()
    columns = set()
    
    # WHERE clause
    where_match = re.search(r'where (.*?)(group by|order by|$)', query)
    if where_match:
        where_clause = where_match.group(1)
        columns.update(re.findall(r'([a-z0-9_]+\.[a-z0-9_]+)', where_clause))
    
    # JOIN clause
    join_matches = re.findall(r'join .*? on (.*?)(where|join|group by|order by|$)', query)
    for jm in join_matches:
        columns.update(re.findall(r'([a-z0-9_]+\.[a-z0-9_]+)', jm[0]))
    
    # GROUP BY clause
    group_match = re.search(r'group by (.*?)(order by|$)', query)
    if group_match:
        group_clause = group_match.group(1)
        columns.update([col.strip() for col in group_clause.split(',')])
    
    # ORDER BY clause
    order_match = re.search(r'order by (.*)', query)
    if order_match:
        order_clause = order_match.group(1)
        columns.update([col.strip() for col in order_clause.split(',')])
    
    # Return as comma-separated string
    return ','.join(columns) if columns else None

# Apply extraction to the query column
df['target_indexes'] = df['query'].apply(extract_index_columns)

# Save the new CSV
df.to_csv('merged_queries_with_indexes.csv', index=False)
print("Generated target_indexes column successfully!")

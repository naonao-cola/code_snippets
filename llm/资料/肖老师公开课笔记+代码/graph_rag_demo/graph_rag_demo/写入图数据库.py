from langchain_neo4j import Neo4jGraph

# 数据的连接
graph_db = Neo4jGraph(
    url='bolt://39.100.64.14/:7687',
    username='neo4j',
    password='1qaz3edc',
    database='neo4j',
    enhanced_schema=True,
)

# 数据库的图结构
schema = graph_db.schema
print(schema)
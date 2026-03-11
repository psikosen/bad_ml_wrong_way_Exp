"""Optional Neo4j property graph store for persistent memory.

Requires a running Neo4j instance and the neo4j Python driver.
This module is entirely optional; experiments work without it.
"""

from typing import Dict, List, Optional

try:
    from neo4j import GraphDatabase

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False


class Neo4jStore:
    """Thin wrapper around Neo4j for MemoryNode/MemoryEdge persistence."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.driver = None
        if HAS_NEO4J:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @property
    def available(self) -> bool:
        return self.driver is not None

    def close(self):
        if self.driver:
            self.driver.close()

    def upsert_node(self, node_id: str, node_type: str, text: str, source_type: str, confidence: float, created_at: str):
        if not self.available:
            return
        with self.driver.session() as session:
            session.run(
                """
                MERGE (n:MemoryNode {node_id: $node_id})
                SET n.node_type = $node_type,
                    n.text = $text,
                    n.source_type = $source_type,
                    n.confidence = $confidence,
                    n.created_at = datetime($created_at)
                """,
                node_id=node_id,
                node_type=node_type,
                text=text,
                source_type=source_type,
                confidence=confidence,
                created_at=created_at,
            )

    def upsert_edge(self, source_id: str, target_id: str, rel_type: str, weight: float, created_at: str):
        if not self.available:
            return
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:MemoryNode {node_id: $source_id}), (b:MemoryNode {node_id: $target_id})
                MERGE (a)-[r:""" + rel_type + """]->(b)
                SET r.weight = $weight, r.created_at = datetime($created_at)
                """,
                source_id=source_id,
                target_id=target_id,
                weight=weight,
                created_at=created_at,
            )

    def query_neighbors(self, node_id: str) -> List[Dict]:
        if not self.available:
            return []
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:MemoryNode {node_id: $node_id})-[r]-(b:MemoryNode)
                RETURN b.node_id AS neighbor_id, type(r) AS rel_type, r.weight AS weight
                """,
                node_id=node_id,
            )
            return [dict(record) for record in result]

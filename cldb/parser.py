from dataclasses import dataclass, field
from typing import List, Optional, Set
import sqlglot
from sqlglot import exp

@dataclass
class QueryFeatures:
    tables: List[str] = field(default_factory=list)
    join_count: int = 0
    join_types: List[str] = field(default_factory=list)
    predicate_columns: List[str] = field(default_factory=list)
    has_group_by: bool = False
    has_order_by: bool = False
    estimated_cardinality: Optional[float] = None

class SQLParser:
    """Parses SQL queries using sqlglot and extracts structural features."""
    
    def parse(self, query: str) -> QueryFeatures:
        try:
            # Parse the query, defaulting to postgres dialect for CLDB
            parsed = sqlglot.parse_one(query, read="postgres")
        except sqlglot.errors.ParseError as e:
            # If parsing fails, return empty features (or raise an exception depending on design)
            # For robustness, we will return default empty features.
            return QueryFeatures()

        features = QueryFeatures()
        
        # Extract tables
        features.tables = [t.name for t in parsed.find_all(exp.Table)]
        
        # Extract joins
        joins = list(parsed.find_all(exp.Join))
        features.join_count = len(joins)
        for join in joins:
            # Join types can be LEFT, RIGHT, FULL, CROSS, INNER (default if not specified)
            side = join.args.get("side")
            kind = join.args.get("kind")
            join_type = "INNER"
            if side:
                join_type = side.upper()
            elif kind:
                join_type = kind.upper()
            features.join_types.append(join_type)
            
        # Extract predicate columns (from WHERE or ON clauses)
        # We look for columns used in conditional expressions (eq, gt, lt, etc.)
        # A simple approximation is finding all columns in the WHERE and JOIN ON clauses
        predicate_cols: Set[str] = set()
        
        where = parsed.args.get("where")
        if where:
            for col in where.find_all(exp.Column):
                predicate_cols.add(col.name)
                
        for join in joins:
            on = join.args.get("on")
            if on:
                for col in on.find_all(exp.Column):
                    predicate_cols.add(col.name)
                    
        features.predicate_columns = list(predicate_cols)
        
        # Check for GROUP BY
        if parsed.args.get("group"):
            features.has_group_by = True
            
        # Check for ORDER BY
        if parsed.args.get("order"):
            features.has_order_by = True
            
        return features

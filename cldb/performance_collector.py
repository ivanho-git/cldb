import time
from sqlalchemy import text
from cldb.metadata_repo import MetadataRepo

class PerformanceCollector:
    """
    Measures query performance (latency) against the live database
    to gather ground-truth feedback after an optimization decision.
    """
    def __init__(self, repo: MetadataRepo):
        self.repo = repo
        
    def measure_latency(self, query: str, runs: int = 3) -> float:
        """
        Executes the query multiple times and returns the average latency in ms.
        Returns float('inf') if the query fails.
        """
        latencies = []
        try:
            with self.repo.engine.connect() as conn:
                for _ in range(runs):
                    start = time.time()
                    conn.execute(text(query)).fetchall()
                    end = time.time()
                    latencies.append((end - start) * 1000)
        except Exception as e:
            # For prototype, we swallow the error and return inf
            print(f"PerformanceCollector Error on query '{query[:30]}...': {e}")
            return float('inf')
            
        if not latencies:
            return float('inf')
            
        return sum(latencies) / len(latencies)

import os

from sqlalchemy import create_engine

# Set DATABASE_URL to switch databases without touching code, e.g.
#   postgresql+psycopg2://user:pass@localhost:5432/stocks   (local Docker)
#   postgresql+psycopg2://user:pass@<rds-endpoint>:5432/stocks  (AWS RDS)
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///stocks.db")

engine = create_engine(DB_URL, pool_pre_ping=True)

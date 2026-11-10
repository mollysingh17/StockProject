import os

from sqlalchemy import create_engine

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://stocky:stocky@localhost:5432/stocky"
)

engine = create_engine(DB_URL, pool_pre_ping=True)
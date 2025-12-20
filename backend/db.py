from sqlalchemy import create_engine

# SQLite 
DB_URL = "sqlite:///stocks.db"

engine = create_engine(DB_URL)
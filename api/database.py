from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy import inspect

# Use a file-based SQLite database
SQLALCHEMY_DATABASE_URL = "sqlite:///./basketball.db"
print("Using database URL:", SQLALCHEMY_DATABASE_URL)

# Create engine with proper configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=True  # Enable SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db():
    """Get database session with proper error handling."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        print(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialize database with proper error handling."""
    try:
        print("Creating database tables...")
        print("Base metadata tables:", Base.metadata.tables.keys())
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
        
        # Verify tables were created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print("Created tables:", tables)
        
        # Verify each table's columns
        for table in tables:
            columns = inspector.get_columns(table)
            print(f"\nTable '{table}' columns:")
            for column in columns:
                print(f"  - {column['name']}: {column['type']}")
        
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        print("Error details:", e.__class__.__name__)
        import traceback
        print("Traceback:", traceback.format_exc())
        raise 
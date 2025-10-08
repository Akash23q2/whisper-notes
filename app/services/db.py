## imports ##
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base
Base = declarative_base()
from typing import Any, Dict, List

## database engine and session ##
from config import DATABASE_URL
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

## methods ##

def create_tables():
    """ Create all tables from Base metadata """
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

def drop_tables():
    """ Drop all tables from Base metadata """
    Base.metadata.drop_all(bind=engine)
    print("Tables dropped successfully.")

def get_db():
    """ Yield a database session """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_record(record: Any):
    """ Add a new record to database """
    db = SessionLocal()
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error adding record: {e}")
        return None
    finally:
        db.close()

def get_record_by_id(model: Any, record_id: int):
    """ Fetch a record by primary key id """
    db = SessionLocal()
    try:
        return db.query(model).filter(model.id == record_id).first()
    except SQLAlchemyError as e:
        print(f"Error fetching record: {e}")
        return None
    finally:
        db.close()

def delete_record_by_id(model: Any, record_id: int):
    """ Delete a record by primary key id """
    db = SessionLocal()
    try:
        obj = db.query(model).filter(model.id == record_id).first()
        if obj:
            db.delete(obj)
            db.commit()
            print(f"Record id={record_id} deleted successfully.")
            return True
        else:
            print(f"Record id={record_id} not found.")
            return False
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error deleting record: {e}")
        return False
    finally:
        db.close()

def query_records(model: Any, filters: Dict = None, limit: int = 10) -> List[Any]:
    """ Query records from a model with optional filters """
    db = SessionLocal()
    try:
        q = db.query(model)
        if filters:
            for attr, val in filters.items():
                q = q.filter(getattr(model, attr) == val)
        return q.limit(limit).all()
    except SQLAlchemyError as e:
        print(f"Error querying records: {e}")
        return []
    finally:
        db.close()

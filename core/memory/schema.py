import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from core.common.const import MemoryConfig

db_directory = os.path.dirname(MemoryConfig.memory_cache_dir)
if not os.path.exists(db_directory):
    os.makedirs(db_directory)

engine = create_engine(f"sqlite:///{MemoryConfig.memory_cache_dir}/memory.db")
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class BaseMemoryModel(Base):  # type: ignore
    __abstract__ = True

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, onupdate=datetime.utcnow
    )

    def serialize(self):
        obj = {}
        for column in self.__table__.columns:
            if type(getattr(self, column.name)) == datetime:
                obj[column.name] = (
                    getattr(self, column.name).isoformat()
                    if getattr(self, column.name)
                    else None
                )
            else:
                obj[column.name] = getattr(self, column.name)

        return obj

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.serialize()}>"


class UserSession(BaseMemoryModel):
    __tablename__ = "user_session"
    id = Column(Integer, primary_key=True)
    user_name = Column(String, nullable=False)


class SystemConfig(BaseMemoryModel):
    __tablename__ = "system_config"
    id = Column(Integer, primary_key=True)
    system_prompt = Column(String, nullable=True)
    llm_model_url_path = Column(String, nullable=True)
    llm_model_local_path = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=True)


class Message(BaseMemoryModel):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    message = Column(String, nullable=False)
    response = Column(String, nullable=False)
    is_bot = Column(Integer, nullable=False)


Base.metadata.create_all(engine)

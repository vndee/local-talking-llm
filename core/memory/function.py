from core.memory.schema import SessionLocal


class CRUD:
    def __init__(self, db):
        self.db = db

    def create(self, obj):
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def get(self, model, id):
        return self.db.query(model).filter(model.id == id).first()

    def get_all(self, model):
        return self.db.query(model).all()

    def update(self):
        self.db.commit()

    def delete(self, obj):
        self.db.delete(obj)
        self.db.commit()


crud = CRUD(SessionLocal())

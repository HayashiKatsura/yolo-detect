from _api.configuration.DatabaseSession import engine as SQL_ENGINE
from sqlalchemy.orm import Session
from _api.models.file import File as FileObject
from _api.configuration.MimeTypes import *
from datetime import datetime
from sqlalchemy import update
class DeleteFiles:
    def __init__(self, file_ids: list):
        self.file_ids = file_ids

    def _delete_files_local_storage(self):
        with Session(SQL_ENGINE) as session:
            try:
                stmt = (
                    update(FileObject)
                    .where(FileObject.id.in_(self.file_ids))  
                    .values(deleted_at=datetime.utcnow())
                )
                
                session.execute(stmt)
                session.commit() 

                return {"code": 200, "msg": "Files deleted successfully", "file_ids": self.file_ids}

            except Exception as e:
                session.rollback() 
                return {"code": 500, "msg": f"Error deleting files: {str(e)}"}
            

    
    def _delete_files_cloud(self):
        pass
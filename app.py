from fastapi import FastAPI
from _api.controller import model as model_router
from _api.controller import file as file_router
from _api.controller import data as data_router

app = FastAPI(title="YOLO Management API", version="0.1.0")

app.include_router(model_router.router)
app.include_router(file_router.router)
app.include_router(data_router.router)

@app.get("/", tags=["Meta"]) 
def root():
    return {"service": "yolo-platform", "version": "0.1.0"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5173,reload=True)
    
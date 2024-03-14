from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from api import router

app = FastAPI(docs_url="/documentation", redoc_url=None)
templates = Jinja2Templates(directory="templates")

origins = [
    "http://localhost",
    "http://localhost:3000",
]

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("midi.html",{"request":request})
    return templates.TemplateResponse("index.html",{"request":request})

app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8200, reload=True)

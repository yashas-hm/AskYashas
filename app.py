from dotenv import load_dotenv
load_dotenv()

from api.endpoints import default, health_check, answer
from api.utils.middleware import SecurityMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)
app.add_middleware(SecurityMiddleware)

app.include_router(default.router)
app.include_router(health_check.router, prefix='/api')
app.include_router(answer.router, prefix='/api')
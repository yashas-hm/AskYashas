from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.model.query_model import QueryModel
from api.utils.llm_pipeline import get_pipeline

router = APIRouter()


@router.post('/prompt')
async def answer_endpoint(query: QueryModel):
    try:
        pipeline = get_pipeline()
        result = pipeline.invoke(query=query.query, history=query.history)
        return JSONResponse(content={"response": result}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
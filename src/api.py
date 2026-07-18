from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
from src.generate_explanation import explain_user_item

app = FastAPI()


class ExplainRequest(BaseModel):
    user_id: int
    item_id: int


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/explain')
def explain(req: ExplainRequest):
    try:
        result = explain_user_item(req.user_id, req.item_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result

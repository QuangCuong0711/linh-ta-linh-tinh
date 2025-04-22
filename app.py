from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import math
import time


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
            
        selected_move = 

        return AIResponse(move=selected_move)
    except Exception as e:
        print(f"Error: {str(e)}")
        if game_state.valid_moves:
            return AIResponse(move=random.choice(game_state.valid_moves))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
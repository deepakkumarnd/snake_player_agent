import random

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class SnakeBoard(BaseModel):
    grid: List[List[int]]


class NextMoveResponse(BaseModel):
    direction: int


@app.get("/")
def root_path():
    return "ok"


@app.post("/snakes/next-move",
          response_model=NextMoveResponse,
          summary='Predict the next move',
          description='Accept the current state and predict the next move for the snake')
def get_move(current_state: SnakeBoard):
    grid = current_state.grid
    print(grid)
    return {"direction": random.randint(1, 4)}


@app.post("/snakes/feedback")
def feedback(current_state: SnakeBoard, reward: int = 0):
    grid = current_state.grid
    return {}

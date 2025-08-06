from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from snake_player import Agent

app = FastAPI()
agent = Agent()


class SnakeBoard(BaseModel):
    grid: List[List[int]]


class GameFeedback(BaseModel):
    previous_grid: List[List[int]]
    current_grid: List[List[int]]
    reward: int
    game_over: bool
    move: int


class NextMoveResponse(BaseModel):
    direction: int


class StatsResponse(BaseModel):
    games: int
    loss_avg: float
    epsilon: float


@app.get("/")
def root_path():
    return "ok"


@app.post("/snakes/next-move",
          response_model=NextMoveResponse,
          summary='Predict the next move',
          description='Accept the current state and predict the next move for the snake')
def get_move(current_state: SnakeBoard):
    grid = current_state.grid
    action = agent.predict(grid)
    move = action + 1  # front end users 1,2,3,4 as moves
    return {"direction": move}


@app.post("/snakes/feedback")
def feedback(feedback_data: GameFeedback):
    state = feedback_data.previous_grid
    next_state = feedback_data.current_grid
    action = feedback_data.move - 1
    agent.feedback(state, action, feedback_data.reward, next_state, feedback_data.game_over)
    return {}


@app.get("/snakes/stats", response_model=StatsResponse, summary='Game stats')
def get_stats():
    return {
        'games': agent.games,
        'loss_avg': agent.loss_avg or 1000000,
        'epsilon': agent.epsilon
    }

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.game_manager import GameManager

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gm = GameManager(
    csv_path="data/disney_villains_akinator_Dataset.csv",
    model_path="villain_dqn_shaped.pth",
    max_questions=10
)
#gm.load_model()

class AnswerBody(BaseModel):
    game_id: str
    answer: str  # "yes" or "no"

class ConfirmBody(BaseModel):
    game_id: str
    correct: bool

@app.post("/game/start")
def start():
    return gm.start_game()

@app.post("/game/answer")
def answer(body: AnswerBody):
    return gm.answer(body.game_id, body.answer)

@app.post("/game/confirm")
def confirm(body: ConfirmBody):
    return gm.confirm(body.game_id, body.correct)

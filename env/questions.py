# questions.py

ATTRIBUTE_QUESTION_MAP = {
    "IsHuman": "Are they usually human (in their normal form)?",
    "FromPrincessMovie": "Are they from a princess movie?",
    "HasMagicPowers": "Do they have magical powers?",
    "IsShapeShifter": "Can they shapeshift or change form?",
    "HasVillainSong": "Do they have a villain song?",
    "HasHorns": "Do they have horns?",
    "WieldsWeapon": "Do they use a weapon?",
    "HasMinions": "Do they have minions or henchmen?",
    "IsFemale": "Is the villain female?",
    "DiesOrDestroyedAtEnd": "Do they get defeated permanently at the end?",
    "IsCreatureOrSpirit": "Are they a supernatural being or spirit?",
    "CastsCurseOrSpell": "Do they cast curses or powerful spells?",
    "IsGroup": "Are they part of a group rather than a single villain?",
    "RedeemedAtEnd": "Do they get redeemed by the end of the story?",
    "AppearsInSequel": "Do they appear in a sequel?",

}

def pretty_question_from_column(col: str) -> str:
    return ATTRIBUTE_QUESTION_MAP.get(col, col)

def question_for_index(env, q_idx: int) -> str:
    col = env.attribute_cols[q_idx]
    return pretty_question_from_column(col)

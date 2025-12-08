# env/villain_env.py

import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any, List


class VillainAkinatorEnv:
    """
    Custom environment for a Disney Villains Akinator-style RL game.

    - At the start of each episode, a secret villain is chosen at random.
    - The agent can:
        * Ask yes/no questions about attributes (actions 0 .. num_questions-1)
        * Make a final guess of the villain (actions num_questions .. num_questions+num_villains-1)
    - Rewards:
        * Asking a question: -1
        * Correct guess: +10
        * Incorrect guess: -10
    """

    # Columns we will try to use as binary attributes (0/1) from the CSV
    ATTRIBUTE_COLUMNS: List[str] = [
        "IsHuman",
        "IsAnimal",
        "HasMagicPowers",
        "TransformsToAnimal",
        "TransformsToHuman",
        "HasHorns",
        "WieldsWeapon",
        "HasMinions",
        "IsFemale",
        "DiesOrDestroyedAtEnd",
    ]

    def __init__(
        self,
        csv_path: str,
        max_questions: int = 10,
        use_main_villains_only: bool = True,
        name_column: str = "Name",
        random_seed: int = 42,
        min_questions_before_guess: int = 0,
        use_reward_shaping: bool =True
    ):
        self.csv_path = csv_path
        self.max_questions = max_questions
        self.random_seed = random_seed
        self.min_questions_before_guess = min_questions_before_guess

        random.seed(random_seed)
        np.random.seed(random_seed)
        df = pd.read_csv(csv_path)

        if name_column not in df.columns:
            name_column = df.columns[0]

        if use_main_villains_only and "IsMainVillain" in df.columns:
            df = df[df["IsMainVillain"] == 1].reset_index(drop=True)

        max_villains = 20
        df = df.head(max_villains).reset_index(drop=True)

        available_attr_cols = [c for c in self.ATTRIBUTE_COLUMNS if c in df.columns]

        if len(available_attr_cols) == 0:
            raise ValueError(
                "No expected attribute columns found in the CSV. "
                "Please ensure your dataset has the binary columns like IsHuman, HasMagicPowers, etc."
            )

        self.attribute_cols = available_attr_cols

        self.villain_names = df[name_column].tolist()

        self.attributes_matrix = (
            df[self.attribute_cols].fillna(0).astype(int).to_numpy()
        )

        self.num_villains, self.num_questions = self.attributes_matrix.shape

        self.use_reward_shaping = use_reward_shaping
        self.shaping_factor = 1.0
        self.candidate_mask = np.ones(self.num_villains, dtype=bool)

        self.num_actions = self.num_questions + self.num_villains

        self.secret_villain_index: int = -1
        self.asked: np.ndarray = np.zeros(self.num_questions, dtype=np.int32)
        self.answers: np.ndarray = np.zeros(self.num_questions, dtype=np.int32)
        self.num_questions_asked: int = 0
        self.done: bool = False

    @property
    def state_dim(self) -> int:
        return 2 * self.num_questions

    @property
    def action_dim(self) -> int:
        return self.num_actions

    def reset(self) -> np.ndarray:
        """
        Start a new episode:
        - Choose a random villain
        - Reset asked/answers
        - Return initial state
        """
        self.secret_villain_index = random.randint(0, self.num_villains - 1)

        self.asked = np.zeros(self.num_questions, dtype=np.int32)
        self.answers = np.zeros(self.num_questions, dtype=np.int32)

        self.num_questions_asked = 0
        self.done = False

        self.candidate_mask = np.ones(self.num_villains, dtype=bool)

        return self._build_state()

    def step( self, action: int ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self._build_state(), 0.0, True, {"warning": "Episode already done"}

        reward = 0.0
        info: Dict[str, Any] = {}

        if 0 <= action < self.num_questions:
            # Asking a question
            question_index = action

            if self.asked[question_index] == 1:
                # Asking the same question again → small penalty
                reward = -1.0
                # 👉 Still count it as a question so we eventually stop
                self.num_questions_asked += 1

            else:
                # Mark question as asked
                self.asked[question_index] = 1

                # Look up true attribute (0/1) of the secret villain
                attribute_value = self.attributes_matrix[self.secret_villain_index, question_index]

                # Convert 0/1 to -1/+1 encoding
                if attribute_value == 1:
                    self.answers[question_index] = 1
                else:
                    self.answers[question_index] = -1

                # Base cost for asking any question
                reward = -1.0
                self.num_questions_asked += 1

                # -------- Reward shaping: bonus if this question reduces candidates --------
                if self.use_reward_shaping:
                    prev_count = int(self.candidate_mask.sum())

                    if self.answers[question_index] == 1:
                        consistent_with_answer = (self.attributes_matrix[:, question_index] == 1)
                    else:
                        consistent_with_answer = (self.attributes_matrix[:, question_index] == 0)

                    self.candidate_mask = self.candidate_mask & consistent_with_answer
                    new_count = int(self.candidate_mask.sum())

                    if new_count < prev_count and prev_count > 0:
                        reduction_fraction = (prev_count - new_count) / prev_count
                        shaping_bonus = self.shaping_factor * reduction_fraction
                        reward += shaping_bonus
                        info["shaping_bonus"] = shaping_bonus
                # ---------------------------------------------------------------------------

            # Check if we hit max questions
            if self.num_questions_asked >= self.max_questions:
                self.done = True

            next_state = self._build_state()
            return next_state, reward, self.done, info

        else:
            # Guessing a villain
            guess_index = action - self.num_questions

            # If agent tries to guess too early...
            if self.num_questions_asked < self.min_questions_before_guess:
                reward = -5.0
                info["early_guess"] = True
                self.num_questions_asked += 1
                if self.num_questions_asked >= self.max_questions:
                    self.done = True
                next_state = self._build_state()
                return next_state, reward, self.done, info

            # Safety: if invalid index, give a strong penalty and end
            if guess_index < 0 or guess_index >= self.num_villains:
                reward = -10.0
                self.done = True
                next_state = self._build_state()
                info["error"] = "Invalid guess index"
                return next_state, reward, self.done, info

            # --- Normal guessing behavior ---
            secret_name = self.villain_names[self.secret_villain_index]
            guessed_name = self.villain_names[guess_index]

            if guess_index == self.secret_villain_index:
                reward = 10.0
                info["correct_guess"] = True
            else:
                reward = -10.0
                info["correct_guess"] = False

            info["secret_villain"] = secret_name
            info["guessed_villain"] = guessed_name

            self.done = True
            next_state = self._build_state()
            return next_state, reward, self.done, info

    def _build_state(self) -> np.ndarray:
        # Convert to float32 for PyTorch convenience later
        asked_float = self.asked.astype(np.float32)
        answers_float = self.answers.astype(np.float32)

        state = np.concatenate([asked_float, answers_float], axis=0)
        return state

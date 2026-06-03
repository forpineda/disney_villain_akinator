import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any, List
#from env.questions import QUESTION_TEXT


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
        "FromPrincessMovie",
        "HasMagicPowers",
        "IsShapeShifter",
        "HasVillainSong",
        "HasHorns",
        "WieldsWeapon",
        "HasMinions",
        "IsFemale",
        "DiesOrDestroyedAtEnd",
        "IsCreatureOrSpirit",
        "CastsCurseOrSpell",
        "IsGroup",
        "AppearsInSequel",
    ]

    def __init__(
        self,
        csv_path: str,
        max_questions: int = 15,
        use_main_villains_only: bool = True,
        name_column: str = "Name",
        random_seed: int = 42,
        min_questions_before_guess: int = 10,
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

        self.question_cost = 0.35          # was effectively 1.0 before
        self.idk_extra_cost = 0.15
        self.shaping_alpha = 1.25          # how much we reward informative questions

        #self.action_dim = self.num_actions

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

        self.candidate_weights = np.ones(self.num_villains, dtype=np.float32)


    @property
    def state_dim(self) -> int:
        return 2 * self.num_questions +2

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
        self.candidate_weights = np.ones(self.num_villains, dtype=np.float32)

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
                reward = -2.0
                next_state = self._build_state()
                return next_state, reward, self.done, {"error": "repeat_question"}

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
                    before = int(self.candidate_mask.sum())

                    if self.answers[question_index] == 1:
                        consistent = (self.attributes_matrix[:, question_index] == 1)
                    else:
                        consistent = (self.attributes_matrix[:, question_index] == 0)

                    self.candidate_mask &= consistent
                    after = int(self.candidate_mask.sum())

                    # normalized info gain shaping
                    if before > 0:
                        safe_after = max(after, 1)
                        info_gain = np.log2(before / safe_after)
                        max_gain = np.log2(max(self.num_villains, 2))
                        norm_gain = info_gain / max_gain  # 0..1

                        shaping_bonus = self.shaping_alpha * norm_gain
                        reward += shaping_bonus
                        info["shaping_bonus"] = shaping_bonus
                        info["before"] = before
                        info["after"] = after
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
        asked_float = self.asked.astype(np.float32)
        answers_float = self.answers.astype(np.float32)

        # how many candidates are still possible?
        remaining_frac = float(self.candidate_mask.sum()) / float(max(self.num_villains, 1))

        # how many questions are left in this episode?
        questions_left_frac = float(self.max_questions - self.num_questions_asked) / float(max(self.max_questions, 1))

        extra = np.array([remaining_frac, questions_left_frac], dtype=np.float32)

        state = np.concatenate([asked_float, answers_float, extra], axis=0)
        return state

    def step_with_user_answer(self, q_idx: int, ans_code: str):
        if self.done:
            return self._build_state(), 0.0, True, {}

        if q_idx < 0 or q_idx >= self.num_questions:
            return self._build_state(), -5.0, self.done, {"error": "Invalid question index"}

        if self.asked[q_idx] == 1:
            return self._build_state(), -2.0, self.done, {"error": "Question already asked"}

        # mark asked
        self.asked[q_idx] = 1
        self.num_questions_asked += 1

        reward = -self.question_cost  # use your tuned cost (0.35)

        before = float(self.candidate_weights.sum())
        col = self.attributes_matrix[:, q_idx].astype(np.int32)  # 0/1

        # Likelihoods
        P = 0.75  # probability for "probably"
        EPS = 1e-8

        if ans_code == "idk":
            self.answers[q_idx] = 0
            reward -= self.idk_extra_cost

        elif ans_code == "yes":
            self.answers[q_idx] = 1
            mult = np.where(col == 1, 0.95, 0.05).astype(np.float32)
            self.candidate_weights *= mult

        elif ans_code == "no":
            self.answers[q_idx] = -1
            mult = np.where(col == 0, 0.95, 0.05).astype(np.float32)
            self.candidate_weights *= mult

        elif ans_code == "prob_yes":
            self.answers[q_idx] = 1
            mult = np.where(col == 1, 0.75, 0.25).astype(np.float32)
            self.candidate_weights *= mult

        elif ans_code == "prob_no":
            self.answers[q_idx] = -1
            mult = np.where(col == 0, 0.75, 0.25).astype(np.float32)
            self.candidate_weights *= mult

        else:
            # unknown answer type -> treat as idk
            self.answers[q_idx] = 0
            reward -= self.idk_extra_cost

        # keep mask in sync (optional but useful)
        max_w = float(self.candidate_weights.max())
        self.candidate_mask = self.candidate_weights >= max_w * 0.25

        after = float(self.candidate_weights.sum())
        remaining = int(self.candidate_mask.sum())

        total = float(self.candidate_weights.sum())
        if total > EPS:
            self.candidate_weights /= total

        # reward shaping: reward proportional reduction in uncertainty
        if self.use_reward_shaping and before > 0:
            # use weight-sum reduction instead of count reduction
            reduction = (before - after) / max(before, EPS)
            reward += self.shaping_alpha * reduction

        if self.num_questions_asked >= self.max_questions:
            self.done = True

        info = {"remaining_candidates": remaining, "before_w": before, "after_w": after}
        return self._build_state(), float(reward), self.done, info


    def reset_user(self) -> np.ndarray:
        self.secret_villain_index = -1

        self.asked = np.zeros(self.num_questions, dtype=np.int32)
        self.answers = np.zeros(self.num_questions, dtype=np.float32)

        self.num_questions_asked = 0
        self.done = False

        self.candidate_mask = np.ones(self.num_villains, dtype=bool)
        self.candidate_weights = np.ones(self.num_villains, dtype=np.float32)
        self.candidate_weights /= self.candidate_weights.sum()

        return self._build_state()

    def valid_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of shape (num_actions,)
        True = action is allowed, False = action is invalid.
        Actions:
        0..num_questions-1 => ask question q
        num_questions..num_questions+num_villains-1 => guess villain
        """
        mask = np.ones(self.num_actions, dtype=bool)

        # Can't ask a question twice
        mask[: self.num_questions] = (self.asked == 0)

        # Can't guess before min questions
        if self.num_questions_asked < self.min_questions_before_guess:
            mask[self.num_questions :] = False

        return mask


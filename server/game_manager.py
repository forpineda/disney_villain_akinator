import uuid
import numpy as np
import torch

from env.villain_env import VillainAkinatorEnv
from agent.dqn import DQN
from env.questions import question_for_index

class GameManager:
    def __init__(self, csv_path: str, model_path: str, max_questions: int = 10):
        self.csv_path = csv_path
        self.model_path = "villain_dqn_baseline_62v_14q_state30.pth"
        self.max_questions = max_questions

        # Env template (we will create a new env per game)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = None
        self.games = {}  # game_id -> dict(state/env/history)

    def load_model(self):
        # Build env EXACTLY like training (same question set, max_questions, filters)
        env = VillainAkinatorEnv(
            csv_path=self.csv_path,
            max_questions=self.max_questions,
            use_main_villains_only=True,          # keep consistent with your dataset filtering
            min_questions_before_guess=6,         # MUST match training
            use_reward_shaping=True,              # doesn't change dims, but keeps behavior consistent
        )

        # These MUST reflect the new state design (asked + answers + 2 extra features)
        state_dim = env.state_dim
        action_dim = env.action_dim

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.policy_net.eval()

    def _pick_action(self, state_np: np.ndarray) -> int:
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def start_game(self):
        game_id = str(uuid.uuid4())

        env = VillainAkinatorEnv(
            csv_path=self.csv_path,
            max_questions=self.max_questions,
            use_main_villains_only=True,
            min_questions_before_guess=10,
        )

        state = env.reset_user()
        history = []

        action = self._choose_next_question(env, state)

        self.games[game_id] = {
            "env": env,
            "state": state,
            "history": history,
            "last_action": action,
            "done": False,
            "last_info": {},
        }

        return self._format_step(game_id)


    def answer(self, game_id: str, user_answer: str):
        g = self.games[game_id]
        env = g["env"]
        action = g["last_action"]

        if g["done"]:
            return self._format_step(game_id)

        # If the model already chose a guess action, answering doesn't apply
        if action >= env.num_questions:
            return self._format_step(game_id)

        q_idx = action
        ua = user_answer.strip().lower()
        if ua in ("yes", "y", "true", "1"):
            ans = 1
            ans_code = "yes"
        elif ua in ("no", "n", "false", "0"):
            ans = -1
            ans_code = "no"
        elif ua in ("probably yes", "prob_yes", "py", "likely yes", "maybe yes"):
            ans = 0.5
            ans_code = "prob_yes"
        elif ua in ("probably no", "prob_no", "pn", "likely no", "maybe no"):
            ans = -0.5
            ans_code = "prob_no"
        else:
            # "idk", "i don't know", "dont know", etc.
            ans = 0
            ans_code = "idk"

        next_state, reward, done, info = env.step_with_user_answer(q_idx, ans_code)

        remaining = int(env.candidate_mask.sum())

        # Better question text source: env.attribute_cols[q_idx]
        # (Your question_for_index can map q_idx -> env.attribute_cols[q_idx] internally)
        question_text = question_for_index(env, q_idx)
        label_map = {
            "yes": "Yes",
            "no": "No",
            "prob_yes": "Probably Yes",
            "prob_no": "Probably No",
            "idk": "I don't know"
        }
        answer_label = label_map[ans_code]

        g["history"].append({
            "type": "question",
            "q_idx": q_idx,
            "question": question_text,
            "answer": answer_label,
            "reward": reward,
            "remaining_candidates": remaining
        })

        g["state"] = next_state
        g["last_info"] = info

        # --- Option B special case: 0 candidates left ---
        # This means the user's answers contradict the dataset attributes.
        # Recover gracefully: reset candidates and continue asking.
        if remaining == 0:
            '''env.candidate_mask = np.ones(env.num_villains, dtype=bool)
            env.candidate_weights = np.ones(env.num_villains, dtype=np.float32)
            env.candidate_weights = env.candidate_weights.sum()
            g["last_info"] = {"error": "No candidates remain (inconsistent answers). Resetting candidates."}

            next_action = self._choose_next_question(env, next_state)
            g["last_action"] = next_action
            g["done"] = False
            return self._format_step(game_id)'''
            guess_idx = self._pick_guess_from_candidates(env)
            g["last_action"] = env.num_questions + guess_idx
            g["done"] = False
            g["last_info"] = {"warning": "No exact candidates remain, guessing best soft match."}
            return self._format_step(game_id)

        # --- deterministic guessing rules ---
        # Only guess early when you're REALLY confident
        if remaining == 1 and env.num_questions_asked >= env.min_questions_before_guess:
            guess_idx = self._pick_guess_from_candidates(env)
            g["last_action"] = env.num_questions + guess_idx
            g["done"] = False
            return self._format_step(game_id)
        
        # HARD RULE: never guess too early
        if env.num_questions_asked < env.min_questions_before_guess:
            next_action = self._choose_next_question(env, next_state)
            g["last_action"] = next_action
            g["done"] = False
            return self._format_step(game_id)

        # If we've hit the question limit, we must guess
        if env.num_questions_asked >= env.max_questions:
            guess_idx = self._pick_guess_from_candidates(env)
            g["last_action"] = env.num_questions + guess_idx
            g["done"] = False
            return self._format_step(game_id)

        # Otherwise keep asking
        next_action = self._choose_next_question(env, next_state)
        g["last_action"] = next_action
        g["done"] = False
        return self._format_step(game_id)


    def confirm(self, game_id: str, correct: bool):
        g = self.games[game_id]
        env = g["env"]

        action = g["last_action"]

        guessed_villain = None
        if action >= env.num_questions:
            guess_idx = action - env.num_questions
            if 0 <= guess_idx < len(env.villain_names):
                guessed_villain = env.villain_names[guess_idx]

        g["done"] = True
        g["last_info"] = {
            "correct_guess": bool(correct),
            "guessed_villain": guessed_villain,
            "secret_villain": None,  # Option B: user never tells us the secret
        }

        # Also store a history event so the UI shows the final moment
        g["history"].append({
            "type": "confirm",
            "guessed_villain": guessed_villain,
            "correct_guess": bool(correct)
        })

        return self._format_step(game_id)

    def _format_step(self, game_id: str):
        g = self.games[game_id]
        env = g["env"]
        action = g["last_action"]
        done = g["done"]
        info = g["last_info"]
        remaining = int(env.candidate_mask.sum())

        # If episode done, show final info
        if done:
            return {
                "game_id": game_id,
                "status": "done",
                "remaining_candidates": remaining,
                "history": g["history"],
                "result": {
                    "secret_villain": info.get("secret_villain"),
                    "guessed_villain": info.get("guessed_villain"),
                    "correct_guess": info.get("correct_guess", False)
                }
            }

        # If action is question
        if action < env.num_questions:
            return {
                "game_id": game_id,
                "status": "asking",
                "remaining_candidates": remaining,
                "question": {
                    "q_idx": action,
                    "text": question_for_index(env, action)
                },
                "history": g["history"]
            }

        # Else action is guess
        guess_idx = action - env.num_questions
        guessed_name = env.villain_names[guess_idx]
        
        return {
            "game_id": game_id,
            "status": "guessing",
            "remaining_candidates": remaining,
            "guess": {
                "villain": guessed_name
            },
            "history": g["history"]
        }
    
    def _pick_question_dqn(self, env, state_np: np.ndarray) -> int:
        state_tensor = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).detach().cpu().numpy()

        # ✅ ONLY question actions
        q_values = q_values[:env.num_questions].copy()

        # ✅ don't repeat asked questions
        q_values[env.asked == 1] = -1e9

        return int(np.argmax(q_values))

    def _pick_question_info_gain(self, env) -> int:
        """
        Choose the not-yet-asked question that best splits the remaining candidates.
        This is Akinator-style and works amazingly for Option B.
        """
        remaining_mask = env.candidate_mask
        remaining_count = int(remaining_mask.sum())

        # If nothing remains (inconsistent answers), fallback to any unasked question
        unasked = np.where(env.asked == 0)[0]
        if remaining_count <= 1 or len(unasked) == 0:
            return int(unasked[0]) if len(unasked) > 0 else 0

        best_q = None
        best_score = -1.0

        for q in unasked:
            col = env.attributes_matrix[:, q]
            # among remaining candidates, how many have attribute 1 vs 0?
            ones = int(((col == 1) & remaining_mask).sum())
            zeros = remaining_count - ones

            # score: we want the split to be as balanced as possible (maximize information gain)
            # a simple proxy is: min(ones, zeros)
            score = min(ones, zeros)

            if score > best_score:
                best_score = score
                best_q = q

        return int(best_q if best_q is not None else unasked[0])

    def _choose_next_question(self, env, state_np: np.ndarray) -> int:
        return self._pick_question_info_gain(env)

    def _pick_guess_from_candidates(self, env) -> int:
        weights = env.candidate_weights

        if weights.sum() <= 1e-8 or np.isnan(weights).any():
            # emergency fallback: use remaining mask if possible
            remaining_idx = np.where(env.candidate_mask)[0]
            if len(remaining_idx) > 0:
                return int(np.random.choice(remaining_idx))
            return 0

        max_w = weights.max()
        top = np.where(np.isclose(weights, max_w))[0]
        return int(np.random.choice(top))


    
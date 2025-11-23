import numpy as np
import pandas as pd
import random

'''
This would be our custom environment class where it would be define
 state representation
 action space (questions + guesses)
 rewards
 episode ending conditions
'''

class VillainAkinatorEnv:
    def __init__(...):
        '''
        METHOD __init__(self, villains_csv_path, max_questions):
        # load villains data
        load table from Data/villains

        # store list of villain names
        self.villain_names = list of names from table

        # store attributes matrix (0/1 values for each villain and each attribute)
        self.attributes_matrix = 2D array [num_villains x num_questions]

        # set numbers
        self.num_villains = number of rows in table
        self.num_questions = number of attribute columns

        # define how many questions allowed per episode
        self.max_questions = max_questions

        # define action space size:
         0 .. num_questions-1  --> ask question i
         num_questions .. num_questions + num_villains - 1 --> guess villain j
        self.num_actions = self.num_questions + self.num_villains

        # initialize some variables that will be set in reset()
        self.secret_villain_index = None
        self.asked = None
        self.answers = None
        self.num_questions_asked = 0
        self.done = False
        '''

    def reset(self):
        '''
        # randomly choose a secret villain
        self.secret_villain_index = random integer in [0, num_villains)

        # reset arrays tracking questions and answers
            asked[i] = 0 if question i not asked yet, 1 if asked
            answers[i] = 0 if not asked yet, +1 for "yes", -1 for "no"
        self.asked = vector of zeros of length num_questions
        self.answers = vector of zeros of length num_questions

        # reset question counter and done flag
        self.num_questions_asked = 0
        self.done = False

        # build initial state representation
        state = build_state_from(self.asked, self.answers)

        RETURN state
        '''

    def step(self, action):
        '''
        IF self.done is True:
            # either raise an error or just call reset internally.
            # For simplicity, we assume episodes should not call step() after done.
            RAISE error or RETURN current_state, 0, True, info

        # Initialize reward to 0
        reward = 0
        info = empty dictionary

        IF action is a "question" action (0 <= action < num_questions):

            question_index = action

            IF self.asked[question_index] == 1:
                Question already asked
                give a small penalty
                reward = -1
                do not increase question count or change answers
            ELSE:
                mark question as asked
                self.asked[question_index] = 1

                look up the true attribute value of the secret villain
                attribute_value = attributes_matrix[secret_villain_index, question_index]
                attribute_value is 0 or 1

                convert to -1 / +1 encoding for answers
                IF attribute_value == 1:
                    self.answers[question_index] = +1
                ELSE:
                    self.answers[question_index] = -1

                asking a question costs something
                reward = -1

                increase count of questions asked
                self.num_questions_asked += 1

            check if we hit maximum number of questions
            IF self.num_questions_asked >= self.max_questions:
                self.done = True

            build next state
            next_state = build_state_from(self.asked, self.answers)

            RETURN next_state, reward, self.done, info


        ELSE:
            otherwise, action is a "guess" action
            guess_index = action - num_questions  # map to villain index

            IF guess_index == self.secret_villain_index:
                Correct guess
                reward = +10
            ELSE:
                Wrong guess
                reward = -10

            episode ends after a guess
            self.done = True

            next state can be the same state or some terminal representation
            next_state = build_state_from(self.asked, self.answers)

            RETURN next_state, reward, self.done, info
        '''

    def build_state_from(self, asked, answers):
        '''
        convert asked and answers vectors to a single flat state vector.
        like = concatenate [asked, answers]

        state_vector = concatenate(asked, answers)

        state_vector length = 2 * num_questions
        Ex: asked = [0, 1, 0], answers = [0, +1, 0]
        state = [0, 1, 0, 0, +1, 0]

        RETURN state_vector
        '''

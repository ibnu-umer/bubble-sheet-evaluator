import random

def create_answers(qn):
    correct_answers = {}
    for i in range(1, qn + 1):
        correct_answers[i] = random.choice(['A', 'B', 'C', 'D'])
    return correct_answers
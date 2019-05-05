import pickle

import numpy as np


def get_submission_sequences(problems):
  with open("data/submissions.pickle", "rb") as f:
    submissions_dict = pickle.load(f)
  cnt = 0
  res = []
  for lst in submissions_dict.values():
    problem_list = list(map(lambda x: x[2], lst))
    tagged_problem_list = list(filter(lambda t: problems[t].tags, problem_list))
    tagged_len, p_len = len(tagged_problem_list), len(problem_list)
    if tagged_len >= 20 and tagged_len / p_len > 0.8:
      cnt += 1
      # print(cnt, tagged_len, p_len, tagged_len / p_len)
      res.append(problem_list)
  return res

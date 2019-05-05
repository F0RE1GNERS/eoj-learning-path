import pickle

import numpy as np

tagstr_to_int = dict()

class Problem():
  def __init__(self, data):
    self.tags = data["tags"]
    self.reward = data["reward"]

  @property
  def vector(self):
    ret = np.zeros((len(tagstr_to_int) + 11))
    r_reward = int(round(self.reward))
    for i in range(-1, 2):
      if 0 <= r_reward + i <= 10:
        ret[len(tagstr_to_int) + i] = 1
    for tag in self.tags:
      ret[tagstr_to_int[tag]] = 1
    return ret


def init_tagstr_to_int(problems):
  global tagstr_to_int
  tag_set = set([tag for p in problems.values() for tag in p.tags])
  for idx, t in enumerate(sorted(list(tag_set))):
    tagstr_to_int[t] = idx


def get_problems_embeddings():
  with open("data/problems.pickle", "rb") as f:
    problems_dict = pickle.load(f)
  problems = {k: Problem(p) for k, p in problems_dict.items()}
  init_tagstr_to_int(problems)
  max_id = max(problems.keys())
  dim = problems[list(problems.keys())[0]].vector.shape[0]
  embedding_matrix = np.zeros((max_id + 1, dim))
  for k, v in problems.items():
    embedding_matrix[k] = v.vector
  return problems, embedding_matrix

from collections import defaultdict
import numpy as np
from random import randint

from scipy import sparse

MAX_V = 3800
START_VAL = 15

X, y = [], []
user = defaultdict(list)
with open("data/data.csv") as f:
  for line in f:
    time, problem, author = map(int, line.strip().split(","))
    if problem not in user[author]:
      user[author].append(problem)
tot_count = 0
for cnt, (key, lst) in enumerate(user.items()):
  start = START_VAL + randint(0, 5)
  if len(lst) >= start:
    print(cnt)
    learned = [0] * MAX_V
    future = [0] * MAX_V
    for x in lst[:start]:
      learned[x] = 1
      future[x] = 0
    for i in range(start, len(lst), 5):
      X.append(np.array(learned))
      for j in range(10):
        if i + j >= len(lst):
          break
        future[lst[i + j]] = 1 - 0.1 * j
      y.append(np.array(future))
      for j in range(5):
        if i + j >= len(lst):
          break
        learned[lst[i + j]] = 1
        future[lst[i + j]] = 0

X = sparse.csr_matrix(X)
y = sparse.csr_matrix(y)
sparse.save_npz("data/train_x.npz", X)
sparse.save_npz("data/train_y.npz", y)

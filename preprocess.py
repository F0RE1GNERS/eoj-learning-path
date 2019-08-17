import pickle

import numpy as np

from spotlight.cross_validation import user_based_train_test_split
from spotlight.interactions import Interactions


def get_all_user_item_list():
    with open("data/submissions.pickle", "rb") as fd:
        submissions = pickle.load(fd)

    user_item_list = []
    for user, lst in submissions.items():
        for timestamp, _, prob in lst:
            user_item_list.append((timestamp, user, prob))
    user_item_list.sort()
    return np.array(user_item_list, dtype=np.int)


def cross_validation_data(params):
    tot = get_all_user_item_list()
    interactions = Interactions(user_ids=tot[:, 1],
                                item_ids=tot[:, 2],
                                timestamps=tot[:, 0],
                                num_items=params["num_problems"])
    random_state = np.random.RandomState(42)
    train, val = user_based_train_test_split(interactions,
                                             random_state=random_state)
    train = train.to_sequence(max_sequence_length=params["max_sequence_length"],
                              min_sequence_length=params["min_sequence_length"],
                              step_size=params["sequence_step_size"])
    val = val.to_sequence(max_sequence_length=params["max_sequence_length"],
                          min_sequence_length=params["min_sequence_length"],
                          step_size=params["sequence_step_size"])
    return train, val


tagstr_to_int = dict()


class Problem(object):
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
    embedding_matrix = np.random.uniform(0, 1, (max_id + 1, dim)).astype(np.float32)
    for k, v in problems.items():
        embedding_matrix[k] = v.vector
    return problems, embedding_matrix

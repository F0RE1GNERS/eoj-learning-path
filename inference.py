import json

import numpy as np
import torch

from model import eoj_next_problem_net
from preprocess import get_problems_embeddings


def load_model():
    return torch.load("checkpoints/model_best.pth.tar")


model = load_model()


def inference(sequence):
    """
    :param sequence: a list of problems
    :return: prediction: a list of problems

    the prediction does not guarantee anything:
    there could be duplicates between prediction and input;
    there could be non-existing problems.
    """
    sequence = np.expand_dims(np.array(sequence), 0)
    return np.argsort(model.predict(sequence)).tolist()

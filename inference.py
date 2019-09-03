import numpy as np
import torch


def load_model():
    loaded = torch.load("checkpoints/model_best.pth.tar",
                        map_location=torch.device('cpu'))
    loaded._use_cuda = False
    return loaded


model = load_model()


def inference(sequence):
    """
    :param sequence: a list of problems
    :return: prediction: a list of problems

    the prediction does not guarantee anything:
    there could be duplicates between prediction and input;
    there could be non-existing problems.
    """
    preproc = list(filter(lambda d: 0 < d < model.num_items, sequence))
    if not preproc:
        preproc = [1]
    sequence = np.expand_dims(np.array(preproc), 0)
    return np.argsort(model.predict(sequence)).tolist()

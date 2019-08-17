import numpy as np
import torch
from nni import env_vars
import nni

from model import eoj_next_problem_net
from preprocess import cross_validation_data, get_problems_embeddings
from spotlight.evaluation import sequence_mrr_score

params = {
    "max_sequence_length": 200,
    "min_sequence_length": 15,
    "sequence_step_size": None,
    "representation": "lstm",
    "loss": "adaptive_hinge",
    "batch_size": 100,
    "learning_rate": 0.01,
    "l2": 0,
    "num_epochs": 80,
    "predefined_embedding": False,
    "predefined_embedding_freeze": False,
    "embedding_dim": 100,
    "num_problems": None,
    "use_cuda": True
}


if env_vars.trial_env_vars.NNI_PLATFORM is not None:
    nni_params = nni.get_next_parameter()
    if nni_params.get("sequence_step_size") == -1:
        nni_params["sequence_step_size"] = None
    params.update(nni_params)


_, item_embedding_layer = get_problems_embeddings()
if params["num_problems"] is None:
    params["num_problems"] = item_embedding_layer.shape[0]
train, val = cross_validation_data(params)


def get_metric():
    return np.mean(sequence_mrr_score(model, val))


def intermediate_callback(epoch_num, epoch_total):
    metric = get_metric()
    print("Epoch {}: metric {}".format(epoch_num, metric))
    if env_vars.trial_env_vars.NNI_PLATFORM is not None:
        if epoch_num == epoch_total:
            nni.report_final_result(metric)
        else:
            nni.report_intermediate_result(metric)


model = eoj_next_problem_net(params, embeddings=item_embedding_layer)
model.fit(train, verbose=True, on_every_n_epochs=intermediate_callback)
if env_vars.trial_env_vars.NNI_PLATFORM is None:
    torch.save(model, "checkpoints/model_best.pth.tar")

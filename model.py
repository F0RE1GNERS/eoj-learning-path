import torch
import torch.nn as nn
import torch.nn.functional as F

from spotlight.sequence.implicit import ImplicitSequenceModel


def eoj_next_problem_net(params, embeddings=None):
    def embedding_layer_func():
        normalized_embeddings = F.normalize(torch.from_numpy(embeddings))
        return nn.Embedding.from_pretrained(normalized_embeddings,
                                            freeze=params["predefined_embedding_freeze"])

    if params["predefined_embedding"]:
        params["embedding_dim"] = embeddings.shape[1]
    return ImplicitSequenceModel(representation=params["representation"],
                                 loss=params["loss"],
                                 batch_size=params["batch_size"],
                                 learning_rate=params["learning_rate"],
                                 l2=params["l2"],
                                 n_iter=params["num_epochs"],
                                 item_embedding_layer=embedding_layer_func if params["predefined_embedding"] else None,
                                 embedding_dim=params["embedding_dim"],
                                 use_cuda=torch.cuda.is_available())

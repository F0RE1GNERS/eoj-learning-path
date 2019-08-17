# Next Problem Recommender

This is a recommender designed for next problem recommendation (on ECNU Online Judge),
based on Spotlight and PyTorch.

I'm no recommendation expert, so this should be a
fair example on how to get recommendation to work in your system with no prior experience.

## Datasets

The dataset is provided in the following way.

We get all the accepted submission records from our online judge (about 1 million records),
export their `timestamps`, `user_ids` and `problem_ids`. This report is fed directly into the 
network for training and evaluation.

Also, tags information and difficulty is exported and leveraged in our "customized embedding layer".
Currently tags are labelled by humans, and difficulties are derived with an equation with almost
no foundation. Nonetheless, empirically, using such information to initialize embedding layers brings no improvement
on performance.

For security reasons, the dataset is not publicly available yet, but we
are working on that.

## Performance

### Quantitative Study

Our best model has reached xxx and xxx on cross validation.

### Qualitative Study

For a learner, a good learning path should:

* Flexible to user behavior
* Cover the knowledge graph
* Step by step, from easy to hard

We will show, qualitatively, that our model perform well, even without the supervision of human-labelled difficulties and tags.

to be added


## Training and Validation

Run `train.py` to do the training. Should have `pytorch` installed beforehand. Besides, you may find `requirements.txt` useful.

We use [NNI](https://github.com/microsoft/nni) to tune the model and try to find the best hyper-parameter.
More details can be found in `hpo` directory. We release our best checkpoint [here](to be added) for reproducibility.

## Deployment

We provide docker image to deploy. All you need to do is to clone this repo and run

```bash
docker-compose up -d
```

This project is designed for deployment, but there are still a few things left to do for full functioning:

* Model update: as a standalone service, it needs to receive new training data to re-train and update the weights.
* QPS check/optimization: it's not yet clear whether our QPS will affect the stability of our web server.

## Future work

to add

## Acknowledge

This project has a built-in spotlight because the official spotlight has old PyTorch dependencies, and even if it's
up-to-date, we still need to do a few modifications to the original code. Code works are done by me and xxx.
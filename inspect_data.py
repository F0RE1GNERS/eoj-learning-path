import seaborn as sns
import matplotlib.pyplot as plt

from problems import get_problems_embeddings
from submissions import get_submission_sequences

problems, embeddings = get_problems_embeddings()
submission_seqs = get_submission_sequences(problems)

sns.set()
len_dist_seq = [len(x) for x in submission_seqs]
sns.distplot(len_dist_seq)
plt.savefig("vis/len_dist.png")

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from progressbar import ProgressBar

from problems import get_problems_embeddings
from submissions import get_submission_sequences


class Net(nn.Module):
  def __init__(self, input_size, output_size, embeddings, hidden_dim=100):
    super().__init__()
    self.input_size = input_size
    self.output_size = output_size
    self.hidden_dim = hidden_dim
    self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=False)
    self.embeddings = nn.Embedding.from_pretrained(embeddings)
    self.linear = nn.Linear(self.hidden_dim, self.output_size)

  def forward(self, input):
    input_embedded = self.embeddings(input).unsqueeze(0)
    hidden = self.lstm(input_embedded)[0].view(-1, self.hidden_dim)[:-1, :]
    dense = F.softmax(self.linear(hidden), dim=1)
    return dense


if __name__ == "__main__":
  device = torch.device("cuda")
  problems, embeddings = get_problems_embeddings()
  submission_seqs = get_submission_sequences(problems)
  net = Net(embeddings.shape[1], embeddings.shape[0], torch.from_numpy(embeddings).float())
  net.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  for epoch in range(200):
    print("Training on epoch %d" % epoch)
    epoch_loss, epoch_acc = 0., 0.
    counter = Counter()
    with ProgressBar(max_value=len(submission_seqs), redirect_stdout=True) as bar:
      for i, data_cpu in enumerate(submission_seqs):
        optimizer.zero_grad()
        data_gpu = torch.tensor(data_cpu).to(device)
        outputs = net(data_gpu)
        label_output = torch.tensor(data_cpu[1:]).to(device)
        loss = criterion(outputs, label_output)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() / label_output.size()[0]
        _, predicted = torch.max(outputs.data, 1)
        for x in predicted:
          counter[x.item()] += 1
        epoch_acc += (predicted == label_output).sum().item() / label_output.size()[0]
        bar.update(i)
    print("Epoch loss: %.9f, Accuracy: %.9f" % (epoch_loss / len(submission_seqs),
                                                epoch_acc / len(submission_seqs)))
    print(counter)

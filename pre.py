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
  problems, embeddings = get_problems_embeddings()
  submission_seqs = get_submission_sequences(problems)
  net = Net(embeddings.shape[1], embeddings.shape[0], torch.from_numpy(embeddings).float())
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)

  for epoch in range(200):
    print("Training on epoch %d" % epoch)
    with ProgressBar(max_value=len(submission_seqs), redirect_stdout=True) as bar:
      epoch_loss, epoch_acc = 0., 0.
      for i, data in enumerate(submission_seqs):
        optimizer.zero_grad()
        outputs = net(torch.tensor(data))
        label_output = torch.tensor(data[1:])
        loss = criterion(outputs, label_output)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() / label_output.size()[0]
        _, predicted = torch.max(outputs.data, 1)
        epoch_acc += (predicted == label_output).sum().item() / label_output.size()[0]
        bar.update(i)
    print("Epoch loss: %.9f, Accuracy: %.9f" % (epoch_loss / len(submission_seqs),
                                                epoch_acc / len(submission_seqs)))

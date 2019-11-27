import os
import math
import torch
import getpass
import datetime
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import get_vocab
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

class UnconditionalLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size=128, keep_logs=True, base_logdir='./logs', tracks=None):
        #Initialize the module constructor
        super(UnconditionalLSTM, self).__init__()

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(3, embed_dim)

        self.lstm = nn.LSTM(2 * embed_dim, hidden_dim)

        self.proj = nn.Linear(hidden_dim, self.vocab_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        if keep_logs:
            user = getpass.getuser().lower()
            date = str(datetime.datetime.now().date())
            time = str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')

            logdir_name = '{}_{}_{}'.format(user, date, time)
            full_logdir = os.path.join(base_logdir, logdir_name)
            if tracks is not None:
                full_logdir += "_tracks={}".format(tracks)
            os.mkdir(full_logdir)

            self.logdir = full_logdir
            self.log_writer = SummaryWriter(full_logdir, flush_secs=100)

    def forward(self, token_ids):
        '''
        Args:
            token_ids: size is (batch_size, sequence_length)
        '''
        batch_size, seq_len = token_ids.shape
        assert seq_len%3 == 0

        token_embeds = self.token_embedding(token_ids)

        # Permute into (seq_len, batch, embed_size)
        token_embeds = token_embeds.permute(1, 0, 2)

        pos_ids = torch.tensor([0, 1, 2]).repeat(batch_size, seq_len//3)
        pos_embeds = self.pos_embedding(pos_ids)
        pos_embeds = pos_embeds.permute(1, 0, 2)

        full_embeds = torch.cat((token_embeds, pos_embeds), dim=2)

        lstm_out, _ = self.lstm(full_embeds)

        projected = self.proj(lstm_out)

        # preds = F.softmax(projected, dim=2)

        return projected

    def fit(self, dataset, batch_size=8, num_epochs=10, save_interval=10000):
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
            num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        global_step = 0
        for idx in range(num_epochs):
            for batch in tqdm(dataloader, desc='Running batches', total=math.ceil(len(dataset)/batch_size)):

                out.to(self.device)
                out = self.forward(batch)

                # The class dimension needs to go in the middle for the CrossEntropyLoss
                out = out.permute(0, 2, 1)

                # And the labes need to be (batch, additional_dims)
                labels = batch.permute(1, 0)

                loss = loss_fn(out, labels)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.log_writer.add_scalar("loss", loss, global_step)
                global_step += 1

                if global_step%save_interval == 0:
                    checkpoint_name = os.path.join(self.logdir, "model_checkpoint_step_{}.pt".format(global_step))
                    torch.save(self.state_dict(), checkpoint_name)

    def generate(self, condition=[225, 48], k=40, temperature=1, length=100):
        batch_size = 1

        self.eval()

        prev = torch.tensor(condition).unsqueeze(0)
        output = prev

        with torch.no_grad():
            for i in tqdm(range(length)):
                logits = self.forward(output)

                # print("Logits shape: ", logits.shape)
                logits[-1][0] /= temperature
                # print("Logits shape: ", logits.shape)

                # Take the last logits, and mask all but the top k
                masked = self.mask_logits(logits[-1], k=k)

                log_probs = F.softmax(masked, dim=1)

                # print("\nMean log probs: ", torch.mean(log_probs[0]))
                # print("Max log prob: ", torch.max(log_probs[0]))
                prev = torch.multinomial(log_probs, num_samples=1)

                output = torch.cat((output, prev), dim=1)

        output = output.cpu().numpy().tolist()[0]

        return [self.vocab[idx] for idx in output]

    def mask_logits(self, logits, k=None):
        if k is None:
            return logits
        else:
            values = torch.topk(logits, k)[0]
            batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * -1e10,
                               logits)

if __name__ == '__main__':
    model = UnconditionalLSTM(embed_dim=100, hidden_dim=100)

    token_ids = torch.tensor([[60, 10, 10, 64, 10, 10, 68, 10, 10]])

    output = model(token_ids)

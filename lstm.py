import os
import math
import torch
import shutil
import getpass
import datetime
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import decode
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

class UnconditionalLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size=128, logdir='test', tracks=None):
        #Initialize the module constructor
        super(UnconditionalLSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(3, embed_dim)

        self.lstm = nn.LSTM(2 * embed_dim, hidden_dim, num_layers=3, dropout=0.3)

        self.proj = nn.Linear(hidden_dim, self.vocab_size)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        if logdir is not None:
            base_logdir='./logs'

            if logdir == 'test':
                full_logdir = os.path.join(base_logdir, 'test')

                if os.path.exists(full_logdir):
                    # Clear out the test directory
                    shutil.rmtree(full_logdir)

                os.mkdir(full_logdir)
            else:
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

        token_embeds = self.token_embedding(token_ids)

        # Permute into (seq_len, batch, embed_size)
        token_embeds = token_embeds.permute(1, 0, 2)

        # The position ids are just 0, 1, and 2 repeated for as long
        # as the sequence length
        pos_ids = torch.tensor([0, 1, 2]).repeat(batch_size, math.ceil(seq_len/3))[:, :seq_len]
        pos_ids = pos_ids.to(self.device)
        pos_embeds = self.pos_embedding(pos_ids)
        pos_embeds = pos_embeds.permute(1, 0, 2)

        full_embeds = torch.cat((token_embeds, pos_embeds), dim=2)

        lstm_out, _ = self.lstm(full_embeds)

        projected = self.proj(lstm_out)

        # preds = F.softmax(projected, dim=2)

        return projected

    def fit(self, dataset, batch_size=8, num_epochs=10, save_interval=10000):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        global_step = 0
        for idx in range(num_epochs):
            with tqdm(dataloader, desc='Running batches', total=math.ceil(len(dataset)/batch_size)) as t:
                for batch in t:

                    # import pdb; pdb.set_trace()
                    batch = batch.to(self.device)

                    inputs, labels = batch[:, :-1], batch[:, 1:]

                    out = self.forward(inputs)

                    # The class dimension needs to go in the middle for the CrossEntropyLoss
                    out = out.permute(0, 2, 1)

                    # And the labels need to be (batch, additional_dims)
                    labels = labels.permute(1, 0)

                    loss = loss_fn(out, labels)
                    t.set_postfix(Loss=loss.item())

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.log_writer.add_scalar("loss", loss, global_step)
                    global_step += 1

                    if global_step%save_interval == 0:
                        checkpoint_name = os.path.join(self.logdir, "model_checkpoint_step_{}.pt".format(global_step))
                        torch.save(self.state_dict(), checkpoint_name)

        # ensure save after fit
        checkpoint_name = os.path.join(self.logdir, "model_checkpoint_step_{}.pt".format(global_step))
        torch.save(self.state_dict(), checkpoint_name)

        generation = self.generate(k=None, length=120, greedy=False)
        print(generation)
        stream = decode(generation)
        stream.write('midi', os.path.join(self.logdir, 'final_sample.mid'))

    def generate(self, condition=[60, 8, 8], k=40, temperature=1, length=100, greedy=False):
        batch_size = 1

        # remove regularization for generation
        self.eval()

        prev = torch.tensor(condition).unsqueeze(0)
        prev = prev.to(self.device)
        output = prev

        with torch.no_grad():
            for i in tqdm(range(length)):

                logits = self.forward(output)
                logits = logits.to(self.device)

                if greedy:
                    prev = torch.argmax(logits[-1][0]).reshape(1, 1)

                else:
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

        return output

    def mask_logits(self, logits, k=None):
        if k is None:
            return logits
        else:
            values = torch.topk(logits, k)[0]
            batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * -1e10,
                               logits)

    def evaluate_sample(self, generation, dataset):


        with torch.no_grad():

            seq_length = len(generation)

            generation = torch.tensor(generation).reshape(1, seq_length)
            # The class dimension needs to go in the middle for the CrossEntropyLoss
            generation_logits = self.forward(generation).permute(0, 2, 1)
            generation_logits = generation_logits[: seq_length,:,:]
            # Get the original seq_length tokens

            # Needs to be (batch, additional_dims)
            labels = dataset[0].reshape(seq_length, 1)

            import pdb; pdb.set_trace()

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(generation_logits, labels)

            return loss


if __name__ == '__main__':
    model = UnconditionalLSTM(embed_dim=100, hidden_dim=100)

    token_ids = torch.tensor([[60, 10, 10, 64, 10, 10, 68, 10, 10]])

    output = model(token_ids)
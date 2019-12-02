import os
import math
import torch
import shutil
import getpass
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from data_utils import decode
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

class UnconditionalLSTM(nn.Module):
    '''
    LOG LEVEL 0: no logs of any kind
    LOG LEVEL 1: write logs to ./logs/debug
    LOG LEVEL 2: write logs to new directory w/ username & time
    '''
    def __init__(self, embed_dim, hidden_dim, num_layers=2, dropout=0.5, vocab_size=128, log_level=0, log_suffix=None):
        #Initialize the module constructor
        super(UnconditionalLSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab_size = vocab_size

        # Encodes the (pitch, dur, adv) tuples
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Encodes the position within each tuple, i.e. [0, 1, 2, 0, 1, 2, ...]
        self.pos_embedding = nn.Embedding(3, embed_dim)

        # NOTE: input dimension is 2 * embed_dim because we have embeddings for both
        # the token IDs and the positional IDs
        self.lstm = nn.LSTM(2 * embed_dim, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.proj = nn.Linear(hidden_dim, self.vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        logdir = None
        if log_level==1:
            logdir = './logs/debug'
            # Clear out the debug directory
            if os.path.exists(logdir):
                shutil.rmtree(logdir)

            os.mkdir(logdir)

        elif log_level==2:
            user = getpass.getuser().lower()
            date = str(datetime.datetime.now().date())
            time = str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')

            logdir_name = '{}_{}_{}'.format(user, date, time)
            logdir = os.path.join('./logs', logdir_name)
            if log_suffix is not None:
                logdir += log_suffix
            os.mkdir(logdir)           
        
            args_string = "Embed dimension: {}" + \
                          "\nHidden dimension: {}" + \
                          "\nNum layers: {}" + \
                          "\nDropout: {}"
            args_string = args_string.format(embed_dim, hidden_dim, num_layers, dropout)

            with open(os.path.join(logdir, 'args.txt'), 'w') as file:
                file.write(args_string)

        self.prepare_logdir(logdir)

    def prepare_logdir(self, logdir=None):
        if logdir is not None:
            self.logdir = logdir
            self.train_sample_dir = os.path.join(self.logdir, 'train_samples')
            self.eval_sample_dir = os.path.join(self.logdir, 'eval_samples')
            self.checkpoints_dir = os.path.join(self.logdir, 'checkpoints')

            os.mkdir(self.train_sample_dir)
            os.mkdir(self.eval_sample_dir)
            os.mkdir(self.checkpoints_dir)

            self.log_writer = SummaryWriter(self.logdir, flush_secs=100)

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

        return projected

    def fit(self, dataset, batch_size=8, num_epochs=10, save_interval=10000):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
            num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        global_step = 0
        for idx in range(num_epochs):
            with tqdm(dataloader, desc='Running batches', total=math.ceil(len(dataset)/batch_size)) as progbar:
                for batch in progbar:

                    batch = batch.to(self.device)

                    inputs, labels = batch[:, :-1], batch[:, 1:]

                    out = self.forward(inputs)

                    # The class dimension needs to go in the middle for the CrossEntropyLoss
                    out = out.permute(0, 2, 1)

                    # And the labels need to be (batch, additional_dims)
                    labels = labels.permute(1, 0)

                    loss = loss_fn(out, labels)
                    progbar.set_postfix(Loss=loss.item())

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.log_writer.add_scalar("loss", loss, global_step)
                    global_step += 1

                    if global_step%save_interval == 0:
                        self.save_checkpoint(global_step, generate_sample=True)

        # ensure save after fit
        self.save_checkpoint(global_step, generate_sample=True)

    def save_checkpoint(self, global_step, generate_sample=False):
        '''
        Saves the model state dict, and will generate a sample if specified
        '''
        checkpoint_name = os.path.join(self.logdir, "model_checkpoint_step_{}.pt".format(global_step))
        torch.save(self.state_dict(), checkpoint_name)

        if generate_sample:
            generation = self.generate(length=120)
            stream = decode(generation)
            stream.write('midi', os.path.join(self.logdir, 'train_sample_checkpoint_step_{}.mid'.format(global_step)))

    def generate(self, condition=[60, 8, 8], k=None, temperature=1, length=100):
        '''
        If 'k' is None: sample over all tokens in vocabulary
        If temperature == 0: perform greedy generation
        '''
        # remove regularization for generation
        self.eval()

        prev = torch.tensor(condition).unsqueeze(0)
        prev = prev.to(self.device)
        output = prev

        with torch.no_grad():
            for i in tqdm(range(length)):

                logits = self.forward(output)
                logits = logits.to(self.device)

                if temperature == 0:
                    prev = torch.argmax(logits[-1][0]).reshape(1, 1)

                else:
                    logits[-1][0] /= temperature

                    # Take the last logits, and mask all but the top k
                    masked = self.mask_logits(logits[-1], k=k)

                    log_probs = F.softmax(masked, dim=1)

                    prev = torch.multinomial(log_probs, num_samples=1)

                output = torch.cat((output, prev), dim=1)
                
        output = output.cpu().numpy().tolist()[0]

        self.train()

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


if __name__ == '__main__':
    model = UnconditionalLSTM(embed_dim=100, hidden_dim=100)

    token_ids = torch.tensor([[60, 10, 10, 64, 10, 10, 68, 10, 10]])

    output = model(token_ids)
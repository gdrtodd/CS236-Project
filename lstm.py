import os
import math
import torch
import getpass
import datetime
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

class BasslineLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, base_logdir='./logs'):
        #Initialize the module constructor
        super(BasslineLSTM, self).__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim, hidden_dim)

        self.proj = nn.Linear(hidden_dim, vocab_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)


        user = getpass.getuser().lower()
        date = str(datetime.datetime.now().date())
        time = str(datetime.datetime.now().time()).split('.')[0].replace(':', '-')

        logdir_name = '{}_{}_{}'.format(user, date, time)
        full_logdir = os.path.join(base_logdir, logdir_name)
        os.mkdir(full_logdir)
    
        self.log_writer = SummaryWriter(full_logdir, flush_secs=100)

    def forward(self, token_ids):
        embeds = self.embedding(token_ids)

        # Permute into (seq_len, batch, input_size)
        embeds = embeds.permute(1, 0, 2)

        lstm_out, _ = self.lstm(embeds)

        projected = self.proj(lstm_out)

        preds = F.softmax(projected, dim=2)

        return preds

    def fit(self, dataset, batch_size=8, num_epochs=10):
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False,
            num_workers=4)

        loss_fn = nn.CrossEntropyLoss()
        global_step = 0
        for idx in range(num_epochs):
            for batch in tqdm(dataloader, desc='Running batches', total=math.ceil(len(dataset)/dataset.seq_len)):
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


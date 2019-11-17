import numpy as np
from lstm import BasslineLSTM
from midi_sequence_dataset import MIDISequenceDataset
from torch.utils.data import DataLoader
from data_utils import get_vocab

if __name__ == '__main__':
    dataset = MIDISequenceDataset(tracks='Bass', seq_len=10)

    lstm = BasslineLSTM(embed_dim=50, hidden_dim=75, vocab_size=len(get_vocab()))

    lstm.fit(dataset)



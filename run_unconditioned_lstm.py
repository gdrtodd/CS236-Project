import argparse
import numpy as np
from lstm import BasslineLSTM
from midi_sequence_dataset import MIDISequenceDataset
from torch.utils.data import DataLoader
from data_utils import get_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', required=True, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--e_dim', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=100)

    args = parser.parse_args()

    tracks = '-'.join(list(args.tracks))
    dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len)

    lstm = BasslineLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, vocab_size=len(get_vocab()))

    lstm.fit(dataset, batch_size=args.batch_size)



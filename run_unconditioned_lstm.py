import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from midi_sequence_dataset import MIDISequenceDataset
from torch.utils.data import DataLoader
from data_utils import get_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default="lakh",
                        choices=["lakh", "maestro", "final-fantasy"])
    parser.add_argument('--tracks', type=str, nargs='+', required=False, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--e_dim', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=100)

    args = parser.parse_args()

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
        dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len)
    else:
        dataset = MIDISequenceDataset(tracks=None, dataset=args.dataset, seq_len=args.seq_len)

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)

    lstm.fit(dataset, batch_size=args.batch_size, num_epochs=args.num_epochs)

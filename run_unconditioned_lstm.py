import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from midi_sequence_dataset import MIDISequenceDataset
from torch.utils.data import DataLoader
from data_utils import get_vocab

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="lakh",
                        choices=["lakh", "maestro", "final-fantasy"])
    parser.add_argument('--tracks', type=str, nargs='+', default=['Piano'],
                        choices=['all', 'Strings','Bass', 'Drums', 'Guitar', 'Piano', None])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=240)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--log_level', type=int, default=2)
    parser.add_argument('--save_interval', type=int, default=20000)
    parser.add_argument('--log_base_dir', type=str, default='./logs')

    args = parser.parse_args()

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
        dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len, partition="train")
    else:
        dataset = MIDISequenceDataset(tracks=None, dataset=args.dataset, seq_len=args.seq_len)

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, num_layers=args.num_layers,
                             dropout=args.dropout, log_level=args.log_level, log_suffix='_tracks={}'.format(tracks),
                             log_base_dir=args.log_base_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)

    lstm.fit(dataset, batch_size=args.batch_size, num_epochs=args.num_epochs, save_interval=args.save_interval)

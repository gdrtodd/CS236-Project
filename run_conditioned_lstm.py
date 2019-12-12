import os
import torch
import argparse
import numpy as np
from lstm import ConditionalLSTM
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
    parser.add_argument('--validation', action='store_true')

    args = parser.parse_args()

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
        dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len, partition="train")
        if args.validation:
            val_dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len, partition="val")
        else:
            val_dataset = None
    else:
        dataset = MIDISequenceDataset(tracks=None, dataset=args.dataset, seq_len=args.seq_len)

    lstm = ConditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, measure_enc_dim=400, num_layers=args.num_layers,
                           dropout=args.dropout, log_level=args.log_level, log_suffix='_tracks={}'.format(tracks),
                           log_base_dir=args.log_base_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)

    cluster_path = "/scratch/users/schlager"
    if os.path.exists(cluster_path):
        print("Loading measure encodings from SCRATCH")
        measure_enc_dir = os.path.join(cluster_path, 'logs/schlager_2019-12-02_00-34-00_tracks=Bass')
    else:
        print("Loading measure encodings from local files")
        measure_enc_dir = './logs/schlager_2019-12-02_00-34-00_tracks=Bass'

    lstm.fit(dataset, batch_size=args.batch_size, num_epochs=args.num_epochs, save_interval=args.save_interval,
             measure_enc_dir=measure_enc_dir, validation_dataset=val_dataset)

"""
Train the conditioned model using user-provided parameters. Saves the model
checkpoints repeatedly during training to `./logs/<unique_descriptive_model_dir>`
although this can be changed (see command-line parameters below).

The measure_enc_dir points to the directory that houses the measure encoding
object used to provide bass-track model conditioning information to the
conditional model.
"""

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
    parser.add_argument('--measure_enc_dir', type=str, default='./data_processed/measure_encodings.pkl')

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

    if os.path.exists(args.measure_enc_dir):
        measure_enc_dir = args.measure_enc_dir
    else:
        raise ValueError("No measure encoding object found at {}. Please generate one using generate_measure_encodings.py".format(args.measure_enc_dir))

    lstm.fit(dataset, batch_size=args.batch_size, num_epochs=args.num_epochs, save_interval=args.save_interval,
             measure_enc_dir=measure_enc_dir, validation_dataset=val_dataset)

"""
File to sample midi files from a provided conditional model.

Use the --bass_logdir and --melody_logdir to point to directories that house
checkpoints for the respective models. Note that --e_dim and --h_dim should
match the parameters used by the model that is being loaded. --bass_temp
and --melody_temp control the temperature of the sample. A temperature of
zero corresponds to greedy sampling. If --k is provided, only the top k
logits are made available for temperature-based sampling.
"""

import os
import time
import tqdm
import glob
import torch
import argparse
import numpy as np
import music21 as m21
from lstm import UnconditionalLSTM, ConditionalLSTM
from data_utils import decode, open_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bass_logdir', type=str, default='logs/example_trained_bass')
    parser.add_argument('--melody_logdir', type=str, default='logs/example_trained_conditional_melody')
    parser.add_argument('--condition', type=int, nargs='+', required=False, default=[60, 8, 8])
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--bass_sample_len', type=int, default=120)
    parser.add_argument('--melody_sample_len', type=int, default=300)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--bass_temp', type=float, default=0.8)
    parser.add_argument('--melody_temp', type=float, default=0.8)
    parser.add_argument('--num_samples', type=int, default=1)

    # NOTE: if --temp == 0, then we perform greedy generation

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nConstructing BASSLINE model...")
    bassline_lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, log_level=0)
    bassline_lstm = bassline_lstm.to(device)

    # if specified, get specific checkpoint
    checkpoint_dir = os.path.join(args.bass_logdir, 'checkpoints')
    if args.ckp:
        full_path = os.path.join(checkpoint_dir, 'model_checkpoint_step_{}.pt'.format(args.ckp))
        num_steps = args.ckp

    # otherwise, get the last checkpoint (alphanumerically sorted)
    else:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

        # model_checkpoint_step_<step_number>.pt --> <step_number>
        step_numbers = np.array(list(map(lambda x: int(x.split(".")[0].split("_")[-1]), checkpoints)))

        sort_order = np.argsort(step_numbers)
        num_steps = step_numbers[sort_order[-1]]

        # gets the checkpoint path with the greatest number of steps
        last_checkpoint_path = checkpoints[sort_order[-1]]
        full_path = last_checkpoint_path

    print("Loading BASSLINE model weights from {}...".format(full_path))
    bassline_lstm.load_state_dict(torch.load(full_path, map_location=device))

    print("\nConstructing MELODY model...")
    melody_lstm = ConditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, measure_enc_dim=args.h_dim, log_level=0)
    melody_lstm = melody_lstm.to(device)

    # if specified, get specific checkpoint
    checkpoint_dir = os.path.join(args.melody_logdir, 'checkpoints')
    if args.ckp:
        full_path = os.path.join(checkpoint_dir, 'model_checkpoint_step_{}.pt'.format(args.ckp))
        num_steps = args.ckp

    # otherwise, get the last checkpoint (alphanumerically sorted)
    else:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

        # model_checkpoint_step_<step_number>.pt --> <step_number>
        step_numbers = np.array(list(map(lambda x: int(x.split(".")[0].split("_")[-1]), checkpoints)))

        sort_order = np.argsort(step_numbers)
        num_steps = step_numbers[sort_order[-1]]

        # gets the checkpoint path with the greatest number of steps
        last_checkpoint_path = checkpoints[sort_order[-1]]
        full_path = last_checkpoint_path

    print("Loading MELODY model model weights from {}...".format(full_path))
    melody_lstm.load_state_dict(torch.load(full_path, map_location=device))

    for i in tqdm.tqdm(range(args.num_samples)):

        bass_out, melody_out = melody_lstm.generate(bassline_model=bassline_lstm, k=args.k, bass_temp=args.bass_temp,
                             bass_length=args.bass_sample_len, melody_temp=args.melody_temp, melody_length=args.melody_sample_len)

        bass_stream = decode(bass_out)
        melody_stream = decode(melody_out)

        combined_stream = m21.stream.Stream()
        bass_part = m21.stream.Part(id='bass')
        bass_part.append(bass_stream)
        melody_part = m21.stream.Part(id='melody')
        melody_part.append(melody_stream)

        combined_stream.insert(0, melody_part)
        combined_stream.insert(0, bass_part)

        # melody_stream.mergeElements(bass_stream)
        # melody_stream.show('midi')

        sample_dir = './generated_samples/conditional'
        bass_sample_dir = "{}_{}_bass.mid".format(sample_dir, len(glob.glob(sample_dir + "*")))
        melody_sample_dir = "{}_{}_melody.mid".format(sample_dir, len(glob.glob(sample_dir + "*")))
        combined_sample_dir = "{}_{}.mid".format(sample_dir, len(glob.glob(sample_dir+"*")))

        # print("Writing sample to {}...".format(sample_dir))
        combined_stream.write('midi', fp=combined_sample_dir)
        bass_stream.write('midi', fp=bass_sample_dir)
        melody_stream.write('midi', fp=melody_sample_dir)

    open_file(combined_sample_dir)

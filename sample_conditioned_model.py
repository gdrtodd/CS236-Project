import os
import time
import glob
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM, ConditionalLSTM
from data_utils import decode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bass_logdir', type=str, default='logs/schlager_2019-12-02_00-34-00_tracks=Bass')
    parser.add_argument('--melody_logdir', type=str, default='logs/schlager_conditional_2019-12-04_11-40-07_tracks=Piano')
    parser.add_argument('--condition', type=int, nargs='+', required=False, default=[60, 8, 8])
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--bass_sample_len', type=int, default=120)
    parser.add_argument('--melody_sample_len', type=int, default=300)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=1)

    # NOTE: if --temp == 0, then we perform greedy generation

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nConstructing BASSLINE model...")
    bassline_lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, log_level=0)

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

    bass_out, melody_out = melody_lstm.generate(bassline_model=bassline_lstm, k=args.k, temperature=args.temp,
                         bass_length=args.bass_sample_len, melody_length=args.melody_sample_len)

    bass_stream = decode(bass_out)
    melody_stream = decode(melody_out)

    melody_stream.mergeElements(bass_stream)
    melody_stream.show('midi')

    

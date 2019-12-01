import os
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from data_utils import decode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--ckp', type=int, required=True)
    parser.add_argument('--e_dim', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--sample_len', type=int, default=120)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=2)

    args = parser.parse_args()

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, keep_logs=False)

    logdir = os.path.join('./logs', args.logdir)

    full_path = os.path.join(logdir, 'model_checkpoint_step_{}.pt'.format(args.ckp))

    lstm.load_state_dict(torch.load(full_path))

    generation = lstm.generate(k=None, length=args.sample_len, temperature=args.temp)
    print("GENERATED SAMPLE: ", generation)
    stream = decode(generation)

    stream.show('midi')
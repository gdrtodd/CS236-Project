import os
import glob
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from midi_sequence_dataset import MIDISequenceDataset
from data_utils import decode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--sample_len', type=int, default=117)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=2)
    parser.add_argument('--greedy', type=bool, default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, keep_logs=False)

    logdir = os.path.join(args.logdir)

    # if specified, get specific checkpoint
    if args.ckp:
        full_path = os.path.join(logdir, 'model_checkpoint_step_{}.pt'.format(args.ckp))

    # otherwise, get the last checkpoint (alphanumerically sorted)
    else:
        checkpoints = glob.glob(os.path.join(logdir, "*.pt"))
        checkpoints.sort()
        last_checkpoint_path = checkpoints[-1]
        full_path = last_checkpoint_path

    print(full_path)

    lstm.load_state_dict(torch.load(full_path, map_location=device))

    generation = lstm.generate(k=None, length=args.sample_len, temperature=args.temp, greedy=args.greedy)

    dataset = MIDISequenceDataset(tracks=None, dataset="final-fantasy", seq_len=args.sample_len+3)
    sample_loss = lstm.evaluate_sample(generation, dataset)

    print("GENERATED SAMPLE: ", generation)
    print("SAMPLE LOSS: ", sample_loss)
    stream = decode(generation)

    stream.write('midi', os.path.join(logdir, 'sample_with_loss={}.mid'.format(sample_loss)))
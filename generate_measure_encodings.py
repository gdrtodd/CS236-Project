import os
import time
import glob
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from midi_sequence_dataset import MIDISequenceDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--condition', type=int, nargs='+', required=False, default=[60, 8, 8])
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--tracks', type=str, nargs='+', required=False, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--dataset', type=str, default="lakh", choices=['lakh', 'maestro', 'final-fantasy'])
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, log_level=0)

    # if specified, get specific checkpoint
    checkpoint_dir = os.path.join(args.logdir, 'checkpoints')
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

    print("Loading model weights from {}...".format(full_path))
    lstm.load_state_dict(torch.load(full_path, map_location=device))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
    else:
        tracks = None

    dataset = MIDISequenceDataset(tracks, dataset=args.dataset)

    lstm.generate_measure_encodings(dataset, args.logdir, batch_size=args.batch_size)

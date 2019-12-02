import os
import time
import glob
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM
from data_utils import decode

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--condition', type=int, nargs='+', required=False, default=[60, 8, 8])
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=100)
    parser.add_argument('--h_dim', type=int, default=100)
    parser.add_argument('--sample_len', type=int, default=240)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=1)

    # NOTE: if --temp == 0, then we perform greedy generation

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, log_level=0)

    # if specified, get specific checkpoint
    if args.ckp:
        full_path = os.path.join(args.logdir, 'model_checkpoint_step_{}.pt'.format(args.ckp))
        num_steps = args.ckp

    # otherwise, get the last checkpoint (alphanumerically sorted)
    else:
        checkpoints = glob.glob(os.path.join(args.logdir, "*.pt"))

        # model_checkpoint_step_<step_number>.pt --> <step_number>
        step_numbers = np.array(list(map(lambda x: int(x.split(".")[0].split("_")[-1]), checkpoints)))

        sort_order = np.argsort(step_numbers)
        num_steps = step_numbers[sort_order[-1]]

        # gets the checkpoint path with the greatest number of steps
        last_checkpoint_path = checkpoints[sort_order[-1]]
        full_path = last_checkpoint_path

    print("Loading model weights from {}...".format(full_path))
    lstm.load_state_dict(torch.load(full_path, map_location=device))

    generation = lstm.generate(condition=args.condition, k=None, length=args.sample_len, temperature=args.temp)

    print("Generated sample: ", generation)
    stream = decode(generation)

    num_eval_samples = len(glob.glob(os.path.join(args.logdir, 'eval_sample*')))

    write_dir = os.path.join(args.logdir, 'eval_sample_checkpoint_{}_{}.mid'.format(str(num_steps), num_eval_samples))

    print("Writing sample to {}...".format(write_dir))
    stream.write('midi', write_dir)

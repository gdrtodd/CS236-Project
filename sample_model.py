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
    parser.add_argument('--sample_len', type=int, default=120)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=2)
    parser.add_argument('--greedy', type=bool, default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim)

    logdir = os.path.join(args.logdir)

    # if specified, get specific checkpoint
    if args.ckp:
        full_path = os.path.join(logdir, 'model_checkpoint_step_{}.pt'.format(args.ckp))
        num_steps = args.ckp

    # otherwise, get the last checkpoint (alphanumerically sorted)
    else:
        checkpoints = glob.glob(os.path.join(logdir, "*.pt"))

        # model_checkpoint_step_<step_number>.pt --> <step_number>
        step_numbers = np.array(list(map(lambda x: int(x.split(".")[0].split("_")[-1]), checkpoints)))
        sort_order = np.argsort(step_numbers)
        num_steps = step_numbers[sort_order[-1]]

        # gets the checkpoint path with the greatest number of steps
        last_checkpoint_path = checkpoints[sort_order[-1]]
        full_path = last_checkpoint_path

    print(full_path)

    lstm.load_state_dict(torch.load(full_path, map_location=device))

    generation = lstm.generate(condition=args.condition, k=None, length=args.sample_len, temperature=args.temp, greedy=args.greedy)

    print("GENERATED SAMPLE: ", generation)
    stream = decode(generation)

    write_dir = os.path.join(logdir, 'sample_steps_{}_{}.mid'.format(str(num_steps), time.strftime("%Y-%m-%d_%H-%M-%S")))

    print("Writing sample to: ", write_dir)

    stream.write('midi', write_dir)
    # stream.show('midi')

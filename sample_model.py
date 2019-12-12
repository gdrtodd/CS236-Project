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
    parser.add_argument('--logdir2', type=str, default='')
    parser.add_argument('--condition', type=int, nargs='+', required=False, default=[60, 8, 8])
    parser.add_argument('--condition2', type=int, nargs='+', default=[36, 8, 8])
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--sample_len', type=int, default=240)
    parser.add_argument('--sample_len2', type=int, default=240)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--temp', type=float, default=1)

    # NOTE: if --temp == 0, then we perform greedy generation

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # GENERATE SAMPLE FROM FIRST LSTM
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

    generation = lstm.generate(condition=args.condition, k=None, length=args.sample_len, temperature=args.temp)
    stream = decode(generation)

    # IF SPECIFIED, SAMPLE FROM SECOND LSTM
    if args.logdir2 is not '':
        lstm2 = UnconditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, log_level=0)

        # if specified, get specific checkpoint
        checkpoint_dir = os.path.join(args.logdir2, 'checkpoints')
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
        lstm2.load_state_dict(torch.load(full_path, map_location=device))

        generation2 = lstm2.generate(condition=args.condition, k=None, length=args.sample_len2, temperature=args.temp)
        stream2 = decode(generation2)

        # COMBINE THE SAMPLES
        stream.mergeElements(stream2)

    print("Finished generating!")
    stream.show('midi')

    write_dir = os.path.join(args.logdir, 'eval_samples')
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)

    num_eval_samples = len(glob.glob(os.path.join(write_dir, 'eval_sample*')))
    output_name = os.path.join(write_dir,
                               'eval_sample_checkpoint_{}_temp_{}_num_{}.mid'.format(str(num_steps),
                                                                                 str(args.temp).replace('.', '-'),
                                                                                 num_eval_samples))

    print("Writing sample to {}...".format(output_name))
    stream.write('midi', output_name)

import os
import time
import glob
import torch
import argparse
import numpy as np
from lstm import UnconditionalLSTM, ConditionalLSTM
from midi_sequence_dataset import MIDISequenceDataset
from data_utils import decode, open_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--partition', type=str, default="test")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=240)
    parser.add_argument('--ckp', type=int, required=False)
    parser.add_argument('--e_dim', type=int, default=200)
    parser.add_argument('--h_dim', type=int, default=400)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--melody_sample_len', type=int, default=240)

    # NOTE: if --temp == 0, then we perform greedy generation

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    tracks = '-'.join(list(["Piano"]))
    dataset = MIDISequenceDataset(tracks=tracks, seq_len=args.seq_len, partition=args.partition)

    print("\nConstructing MELODY model...")
    lstm = ConditionalLSTM(embed_dim=args.e_dim, hidden_dim=args.h_dim, measure_enc_dim=400, num_layers=args.num_layers,
                           log_level=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm.to(device)

    cluster_path = "/scratch/users/schlager"
    if os.path.exists(cluster_path):
        print("Loading measure encodings from SCRATCH")
        measure_enc_dir = os.path.join(cluster_path, 'logs/schlager_2019-12-02_00-34-00_tracks=Bass')
    else:
        print("Loading measure encodings from local files")
        measure_enc_dir = './logs/schlager_2019-12-02_00-34-00_tracks=Bass'

    print("Loading MELODY model model weights from {}...".format(full_path))
    lstm.load_state_dict(torch.load(full_path, map_location=device))

    mean_loss = lstm.evaluate(dataset, batch_size=args.batch_size, measure_enc_dir=measure_enc_dir)

    print("MEAN LOSS: ", mean_loss)




    

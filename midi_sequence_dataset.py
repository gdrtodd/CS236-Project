import os
import torch
import argparse
import numpy as np
import music21 as m21
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import encode, get_vocab
from multiprocessing import Pool


class MIDISequenceDataset(Dataset):
    def __init__(self, tracks, seq_len=120, num_threads=4, cache_dir='./data_processed/', dataset="lakh"):
        # The sequence length needs to be divisible by 3 so that the positional encodings
        # line up properly
        assert seq_len%3 == 0
        self.seq_len = seq_len

        if dataset == "lakh":
            self.data_dir = os.path.join(cache_dir, 'midis_tracks={}'.format(tracks))
            self.save_dir = os.path.join(cache_dir, "token_dataset_tracks={}".format(tracks))
        else:  # dataset == "maestro"
            self.data_dir = os.path.join(cache_dir, '{}_tracks'.format(dataset))
            self.save_dir = os.path.join(cache_dir, "token_dataset_{}".format(dataset))

        if not os.path.exists(self.save_dir):
            print("No token cache found, parsing MIDI files from {} ...".format(self.data_dir))

            token_ids = []

            midis = os.listdir(self.data_dir)

            skip_count = 0

            if num_threads > 1:
                with Pool(num_threads) as pool:
                    ids_by_midi = list(tqdm(pool.imap(self.midi_to_token_ids, midis),
                                            desc='Encoding MIDI streams', total=len(midis)))

                token_ids = []
                for ids in tqdm(ids_by_midi, desc='Adding MIDIs to main encoding', total=len(ids_by_midi)):
                    token_ids += ids

            else:
                for midi_name in tqdm(midis, desc='Encoding MIDI streams', total=len(midis)):
                    path = os.path.join(self.data_dir, midi_name)

                    try:
                        stream = m21.converter.parse(path)
                    except:
                        print("Skipping {} because it took too long to parse".format(midi_name))
                        skip_count += 1
                        continue

                    token_id_encoding = encode(stream)

                    token_ids += token_id_encoding

                print("\nSkipped {} out of {} files".format(skip_count, len(midis)))

            self.token_ids = np.array(token_ids, dtype=np.uint16)

            with open(self.save_dir, 'wb') as file:
                np.save(file, self.token_ids)

        else:
            print("Loading token cache from {} ...".format(self.save_dir))
            with open(self.save_dir, 'rb') as file:
                self.token_ids = np.load(file)

    def midi_to_token_ids(self, midi_name):
        path = os.path.join(self.data_dir, midi_name)
        try:
            stream = m21.converter.parse(path)
            encoding = encode(stream)

            return encoding
        except:
            return []


    def __len__(self):
        return len(self.token_ids)//self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return torch.LongTensor(self.token_ids[start:start+self.seq_len].astype(np.double))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', required=False, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--dataset', type=str, required=True, choices=['lakh', 'maestro', 'final-fantasy'])
    parser.add_argument('--threads', type=int, required=False, default=4)

    args = parser.parse_args()

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
    else:
        tracks = None

    dataset = MIDISequenceDataset(tracks, num_threads=args.threads, dataset=args.dataset)
    

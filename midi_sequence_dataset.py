import os
import torch
import argparse
import numpy as np
import music21 as m21
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import encode, get_vocab


class MIDISequenceDataset(Dataset):
    def __init__(self, tracks, seq_len=50, cache_dir='./data_processed/'):
        self.seq_len = seq_len

        data_dir = os.path.join(cache_dir, 'midis_tracks={}'.format(tracks))
        save_dir = os.path.join(cache_dir, "token_dataset_tracks={}".format(tracks))

        if not os.path.exists(data_dir):
            print("No token cache found, parsing MIDI files from {} ...".format(data_dir))

            token_ids = []

            midis = os.listdir(data_dir)

            skip_count = 0
            for midi_name in tqdm(midis, desc='Parsing MIDIs', total=len(midis)):
                path = os.path.join(data_dir, midi_name)

                try:
                    stream = m21.converter.parse(path)
                except:
                    skip_count += 1
                    continue

                string_encoding = encode(stream)
                token_id_encoding = [vocab.index(token) for token in string_encoding]

                token_ids += token_id_encoding

            print("\nSkipped {} out of {} files".format(skip_count, len(midis)))

            self.token_ids = np.array(token_ids, dtype=np.uint16)

            with open(save_dir, 'wb') as file:
                np.save(file, self.token_ids)

        else:
            print("Loading token cache from {} ...".format(save_dir))
            with open(save_dir, 'rb') as file:
                self.token_ids = np.load(file)

    def __len__(self):
        return len(self.token_ids)//self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return torch.LongTensor(self.token_ids[start:start+self.seq_len].astype(np.double))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', required=True, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])

    args = parser.parse_args()

    tracks = '-'.join(list(args.tracks))

    dataset = MIDISequenceDataset(tracks)
    
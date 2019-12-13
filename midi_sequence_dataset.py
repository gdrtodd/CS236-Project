"""
MIDISequenceDataset is used to parse MIDI sequences in a song dataset
into the processed, tokenized form for language model fitting. It also
serves as the "server-side" dataloader for loading these tokenized data
into the model during training.

Parsing a dataset example:
`python midi_sequence_dataset.py \
--dataset lakh \
--tracks Piano \
--threads 4`
"""

import os
import glob
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import music21 as m21
import multiprocessing
from tqdm import tqdm
from torch.utils.data import Dataset
from data_utils import encode, get_vocab
from multiprocessing import Pool


class MIDISequenceDataset(Dataset):
    def __init__(self, tracks, seq_len=120, num_threads=4, cache_dir='./data_processed/', dataset="lakh",
                 partition="train"):
        # The sequence length needs to be divisible by 3 so that the positional encodings
        # line up properly
        assert seq_len%3 == 0
        self.seq_len = seq_len

        if dataset == "lakh":
            self.data_dir = os.path.join(cache_dir, 'midis_tracks={}'.format(tracks))
            self.save_dir = os.path.join(cache_dir, "token_dataset_tracks={}_{}".format(tracks, partition))

            # Use whole dataset (unpartitioned)
            if partition is None:
                self.save_dir = os.path.join(cache_dir, "token_dataset_tracks={}".format(tracks))

            self.lookup_file = os.path.join(cache_dir, "bass_piano_track_lookup")
        else:  # dataset == "maestro"
            self.data_dir = os.path.join(cache_dir, '{}_tracks'.format(dataset))
            self.save_dir = os.path.join(cache_dir, "token_dataset_{}".format(dataset))

        with open(self.lookup_file, "rb") as f:
            self.lookup_table = pickle.load(f)

        # If tokenized dataset does not exist, create it by processing
        # the provided MIDI files (in `self.data_dir`). Saves the processed
        # token object to `self.save_dir`.
        if not os.path.exists(self.save_dir):
            print("No token cache found, parsing MIDI files from {} ...".format(self.data_dir))

            token_ids = []

            midis = os.listdir(self.data_dir)

            skip_count = 0

            all_token_ids = []
            all_measure_ids = []
            all_track_ids = []

            if num_threads > 1:
                with Pool(num_threads) as pool:
                    # Each entry in this list is of the form [token_ids, measure_ids], where
                    # 1. token_ids: is a list of 3-tuples encoding the midi
                    # 2. measure_ids: is a list of the measure index for each note value
                    info_by_midi = list(tqdm(pool.imap(self.midi_to_token_ids, midis),
                                             desc='Encoding MIDI streams', total=len(midis)))

                for token_ids, measure_ids, track_ids in tqdm(info_by_midi, desc='Adding MIDIs to main encoding',
                                                              total=len(info_by_midi)):
                    all_token_ids += token_ids
                    all_measure_ids += measure_ids
                    all_track_ids += track_ids

            else:
                for midi_name in tqdm(midis, desc='Encoding MIDI streams', total=len(midis)):

                    info_by_midi = self.midi_to_token_ids(midi_name)

                    if info_by_midi == [[], [], []]:
                        skip_count += 1
                        continue

                    else:
                        token_ids, measure_ids, track_ids = info_by_midi

                    all_token_ids += token_ids
                    all_measure_ids += measure_ids
                    all_track_ids += track_ids

                print("\nSkipped {} out of {} files".format(skip_count, len(midis)))

            self.token_ids = np.array(all_token_ids, dtype=np.uint16)
            self.measure_ids = np.array(all_measure_ids, dtype=np.uint16)
            self.track_ids = np.array(all_track_ids, dtype=np.uint16)

            with open(self.save_dir, 'wb') as file:
                np.savez(file, token_ids=self.token_ids, measure_ids=self.measure_ids, track_ids=self.track_ids)

        # If tokenized data exists (numpy object), load in the important
        # info (token_ids [MIDI encoding], measure_ids [to keep track of
        # location when training conditional model], and track_ids [to keep
        # track of which song is being parsed when training conditional
        # model]).
        else:
            print("Loading token cache from {} ...".format(self.save_dir))
            with open(self.save_dir, 'rb') as file:
                dataset_files = np.load(file)
                self.token_ids = dataset_files["token_ids"]
                self.measure_ids = dataset_files["measure_ids"]
                self.track_ids = dataset_files["track_ids"]


    def midi_to_token_ids(self, midi_name):
        """
        Helper function to encode midis from a dataset. See
        data_utils.py's encode() for the specifics.
        """
        path = os.path.join(self.data_dir, midi_name)
        try:
            stream = m21.converter.parse(path)
            encoding, measures = encode(stream)

            track_id = self.get_track_id(path)
            track_ids = [track_id] * len(encoding)

            return encoding, measures, track_ids
        except:
            return [[], [], []]

    def get_track_id(self, midi_path):
        """
        Get the unique track id number from the dataset
        lookup table.
        """

        track_id_seq = self.get_track_id_seq(midi_path)
        return self.lookup_table[track_id_seq]

    def get_track_id_seq(self, track_file_path):
        """
        Given a file path string, returns the track id seq for lookup. Assumes
        filename format is lakh midi style
        :param track_file_path: path to a .mid file
        :return: track id seq (e.g. TRAAAGR128F425B14B)
        """
        track_file = os.path.basename(track_file_path)
        track_id_seq = track_file.split('-')[0]

        return track_id_seq


    def __len__(self):
        return len(self.token_ids)//self.seq_len

    def __getitem__(self, idx):
        """
        Method called by torch.dataloader object during training.
        """
        start = idx * self.seq_len

        return (torch.LongTensor(self.token_ids[start:start+self.seq_len].astype(np.double)),
                torch.LongTensor(self.measure_ids[start:start+self.seq_len].astype(np.double)),
                torch.LongTensor(self.track_ids[start:start+self.seq_len].astype(np.double)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', required=False, choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])
    parser.add_argument('--dataset', type=str, default="lakh", choices=['lakh', 'maestro', 'final-fantasy'])
    parser.add_argument('--threads', type=int, required=False, default=4)

    args = parser.parse_args()

    if args.dataset == "lakh":
        tracks = '-'.join(list(args.tracks))
    else:
        tracks = None

    dataset = MIDISequenceDataset(tracks, num_threads=args.threads, dataset=args.dataset)

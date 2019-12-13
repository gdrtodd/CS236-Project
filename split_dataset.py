"""
Splits a dataset into train, val, and test partitions. Saves these partitions
as the same tokenized file objects as the original dataset, with the name of
the partition appended to the filename.

Example: token_dataset_tracks=Bass --> token_dataset_tracks=Bass_train , _val, _test
"""

import os
import argparse
import pickle
import random
from tqdm import tqdm
import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLE_PATH = os.path.join(PATH,
        "data_processed/bass_piano_track_lookup")
OUTPUT_TRAIN = os.path.join(PATH,
        "data_processed/data_splits/train.pickle")
OUTPUT_TEST = os.path.join(PATH,
        "data_processed/data_splits/test.pickle")
OUTPUT_VAL = os.path.join(PATH,
        "data_processed/data_splits/val.pickle")

def get_track_lookup_dict():
    track_lookup = None
    with open(LOOKUP_TABLE_PATH, 'rb') as fp:
        track_lookup = pickle.load(fp)

    return track_lookup

def partition_dataset(track_ids, train_split=80, test_split=10, val_split=10):
    assert(train_split + test_split + val_split == 100)

    random.shuffle(track_ids)
    size = len(track_ids)

    train = track_ids[:(size // 100) * train_split]
    val = track_ids[(size // 100) * train_split : (size // 100) *
            (train_split + val_split)]
    test = track_ids[(size // 100) * (train_split + val_split) :]
    
    return(train, val, test)

def create_split_datasets(dataset_file, train, val, test):

    with open(dataset_file, 'rb') as file:
        dataset = np.load(file)
        token_ids = dataset["token_ids"]
        measure_ids = dataset["measure_ids"]
        track_ids = dataset["track_ids"]

    for i, id_set in tqdm(enumerate([train, val, test])):
        subset_type = ["train", "val", "test"][i]
        selection = np.where(np.isin(track_ids, id_set))

        subset_token_ids = token_ids[selection]
        subset_measure_ids = measure_ids[selection]
        subset_track_ids = track_ids[selection]

        with open("{}_{}".format(dataset_file, subset_type), 'wb') as file:
            np.savez(file, token_ids=subset_token_ids, measure_ids=subset_measure_ids,
                     track_ids=subset_track_ids)

def main(args):
    track_lookup = get_track_lookup_dict()
    track_ids = list(track_lookup.values()) # Use just the id number
    train, val, test = partition_dataset(track_ids, args.train_split, 
            args.test_split, args.val_split)

    create_split_datasets("data_processed/token_dataset_tracks=Bass", train, val, test)
    create_split_datasets("data_processed/token_dataset_tracks=Piano", train, val, test)

    # Save all the partitions to file.
    with open(OUTPUT_TRAIN, 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUTPUT_VAL, 'wb') as handle:
        pickle.dump(val, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(OUTPUT_TEST, 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Split dataset into', len(train),  'training examples,', len(test),
    'test examples, and', len(val), 'validation examples.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_split', type=int, default=80)
    parser.add_argument('--test_split', type=int, default=10)
    parser.add_argument('--val_split', type=int, default=10)
    args = parser.parse_args()

    main(args)

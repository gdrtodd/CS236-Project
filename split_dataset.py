import os
import argparse
import pickle
import random

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

def main(args):
    track_lookup = get_track_lookup_dict()
    track_ids = list(track_lookup.keys()) # Use just the track ids.
    train, val, test = partition_dataset(track_ids, args.train_split, 
            args.test_split, args.val_split)

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

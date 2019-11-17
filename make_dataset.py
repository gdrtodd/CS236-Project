import os
import argparse
import numpy as np
import music21 as m21
from tqdm import tqdm
from data_utils import encode, get_vocab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', default='all', choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])

    args = parser.parse_args()

    tracks = '-'.join(list(args.tracks))

    midi_dir = './data_processed/midis_tracks={}'.format(tracks)

    vocab = get_vocab()

    dataset = []

    midis = os.listdir(midi_dir)

    skip_count = 0
    for midi_name in tqdm(midis, desc='Parsing MIDIs', total=len(midis)):
        path = os.path.join(midi_dir, midi_name)

        try:
            stream = m21.converter.parse(path)
        except:
            skip_count += 1
            continue

        string_encoding = encode(stream)

        token_id_encoding = [vocab.index(token) for token in string_encoding]

        dataset += token_id_encoding

    print("\nSkipped {} out of {} files".format(skip_count, len(midis)))

    dataset = np.array(dataset, dtype=np.uint16)

    with open("./data_processed/token_dataset_tracks={}".format(tracks), 'wb') as file:
        np.save(file, dataset)
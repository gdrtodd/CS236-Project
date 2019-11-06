import os
import argparse
from pathlib import Path
from tqdm import tqdm
import pypianoroll
from pypianoroll import Multitrack
from tqdm import tqdm

def collect_midis(base_dir, collection_dir, selected_tracks=["all"]):
    """
    Collects .npz files from raw data into processed data folders as .mid
    - selected_track should be a list of track(s) 
        > options: ['Drums', 'Piano', 'Guitar', 'Bass', 'Strings', 'all']
    """

    if not os.path.exists(collection_dir):
        print("Creating collection directory %s" % collection_dir)
        os.mkdir(collection_dir)

    selected_tracks.sort()  # to keep consistency in filename later

    # Find all of the track name directories
    track_paths = list(Path(base_dir).rglob('TR*'))

    for path in tqdm(track_paths, desc='Collecting MIDI files', total=len(track_paths)):
        for checksum in os.listdir(path):
            load_dir = os.path.join(path, checksum)
            multiroll = Multitrack(load_dir)

            # Remove all tracks but those in selected_tracks
            if "all" not in selected_tracks:

                to_remove = [idx for idx, track in enumerate(multiroll.tracks) \
                                if track.name not in selected_tracks]
                multiroll.remove_tracks(to_remove)

                # Make sure our selected tracks persist
                assert len(multiroll.tracks) == len(selected_tracks)

            # e.g. save_name = TR#########-bass-piano.mid
            name = os.path.basename(path)
            save_name = '{}-{}.mid'.format(name, "-".join(selected_tracks).lower())
            save_path = os.path.join(collection_dir, save_name)
            multiroll.write(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracks', type=str, nargs='+', default='all', choices=['all', 'Strings',
                        'Bass', 'Drums', 'Guitar', 'Piano'])

    args = parser.parse_args()
    if args.tracks == 'all':
        args.tracks = ['all']
    
    base_data_dir = './data_raw/lpd_5_cleansed'
    base_collection_dir = './data_processed/'
    full_collection_dir = os.path.join(base_collection_dir, 'midis_tracks=' + '-'.join(args.tracks))

    print("Collecting MIDI files (tracks = {})".format(args.tracks))
    collect_midis(base_data_dir, full_collection_dir, args.tracks)

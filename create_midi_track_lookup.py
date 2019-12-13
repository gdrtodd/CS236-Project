"""
Creates a lookup table for the midi dataset with a unique
track ID for each song. This lookup table is used to uniquely
identify tracks when training the conditional melody model.
"""

import pandas as pd
import pickle
import glob
import os

def get_track_id_seq(track_file_path):
    """
    Given a file path string, returns the track id seq for lookup. Assumes
    filename format is lakh midi style
    :param track_file_path: path to a .mid file
    :return: track id seq (e.g. TRAAAGR128F425B14B)
    """
    track_file = os.path.basename(track_file_path)
    track_id_seq = track_file.split('-')[0]

    return track_id_seq

def create_individual_lookup_for(midi_dir):
    """
    Creates a lookup objection (dict) for a given
    midi directory. Keys are the midi track names
    and values are the unique ids assigned to each.
    """

    midi_track_paths = glob.glob(os.path.join(midi_dir, "*.mid"))
    midi_track_names = [get_track_id_seq(x) for x in midi_track_paths]

    ids, uniques = pd.factorize(midi_track_names)
    lookup = dict(zip(midi_track_names, ids))

    return lookup

if __name__ == "__main__":

    base_dir = "data_processed"

    bass_midi = os.path.join(base_dir, "midis_tracks=Bass")
    piano_midi = os.path.join(base_dir, "midis_tracks=Piano")

    bass_lookup = create_individual_lookup_for(bass_midi)
    piano_lookup = create_individual_lookup_for(piano_midi)

    assert bass_lookup == piano_lookup

    with open(os.path.join(base_dir, "bass_piano_track_lookup"), "wb") as f:
        pickle.dump(bass_lookup, f)

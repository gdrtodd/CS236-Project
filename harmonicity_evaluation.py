"""
Evaluate the harmonicity between bass and piano tracks for different
datasets. Uses code adapted from pypianoroll.
"""

import glob, os
from pypianoroll import metrics, Multitrack
import numpy as np
import warnings

# <Pypianoroll code>

def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance [1].
    Default argument values are set as suggested by the paper.
    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting harmonic
    change in musical audio. In Proc. ACM MM Workshop on Audio and Music
    Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix

def get_num_pitch_used(pianoroll):
    """Return the number of unique pitches used in a piano-roll."""
    return np.sum(np.sum(pianoroll, 0) > 0)

def get_qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    padded = np.pad(pianoroll.astype(int), ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1,)
    onsets = (flattened > 0).nonzero()[0]
    offsets = (flattened < 0).nonzero()[0]
    num_qualified_note = (offsets - onsets >= threshold).sum()
    return num_qualified_note / len(onsets)

def get_polyphonic_ratio(pianoroll, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps"""
    return np.sum(np.sum(pianoroll, 1) >= threshold) / pianoroll.shape[0]

def get_in_scale(chroma, scale_mask=None):
    """Return the ratio of chroma."""
    measure_chroma = np.sum(chroma, axis=0)
    in_scale = np.sum(np.multiply(measure_chroma, scale_mask, dtype=float))
    return in_scale / np.sum(chroma)

def get_drum_pattern(measure, drum_filter):
    """Return the drum_pattern metric value."""
    padded = np.pad(measure, ((1, 0), (0, 0)), 'constant')
    measure = np.diff(padded, axis=0)
    measure[measure < 0] = 0

    max_score = 0
    for i in range(6):
        cdf = np.roll(drum_filter, i)
        score = np.sum(np.multiply(cdf, np.sum(measure, 1)))
        if score > max_score:
            max_score = score

    return  max_score / np.sum(measure)

def get_harmonicity(bar_chroma1, bar_chroma2, resolution, tonal_matrix=None):
    """Return the harmonicity metric value"""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    score_list = []
    for r in range(bar_chroma1.shape[0]//resolution):
        start = r * resolution
        end = (r + 1) * resolution
        beat_chroma1 = np.sum(bar_chroma1[start:end], 0)
        beat_chroma2 = np.sum(bar_chroma2[start:end], 0)
        score_list.append(tonal_dist(beat_chroma1, beat_chroma2, tonal_matrix))
    score_list = np.array(score_list)
    return np.mean(score_list[~np.isnan(score_list)])

def to_chroma(pianoroll):
    """Return the chroma features (not normalized)."""
    padded = np.pad(pianoroll, ((0, 0), (0, 12 - pianoroll.shape[1] % 12)),
                    'constant')
    return np.sum(np.reshape(padded, (pianoroll.shape[0], 12, -1)), 2)

def tonal_dist(chroma1, chroma2, tonal_matrix=None):
    """Return the tonal distance between two chroma features."""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chroma1 = chroma1 / np.sum(chroma1)
        chroma2 = chroma2 / np.sum(chroma2)
    result1 = np.matmul(tonal_matrix, chroma1)
    result2 = np.matmul(tonal_matrix, chroma2)
    return np.linalg.norm(result1 - result2)

# </Pypianoroll code>

def get_dataset_metrics(dir="./generated_samples", base_term="unconditional"):
    """
    Get harmonicity metric for a dataset provided by a directory and base_term.
    Assumes samples are of the form <base_term>_<index>_<bass/melody>.mid.
    """

    bass_metrics = np.zeros((100, 3))
    melody_metrics = np.zeros((100, 3))

    harmonicity = np.zeros(100)

    tonal_matrix = get_tonal_matrix()

    for array_index, track_index in enumerate(range(0, 300, 3)):

        bass_track = Multitrack(os.path.join(dir, "{}_{}_bass.mid".format(base_term, track_index)))
        melody_track = Multitrack(os.path.join(dir, "{}_{}_melody.mid".format(base_term, track_index)))

        bass_pianoroll = bass_track.tracks[0].pianoroll
        melody_pianoroll = melody_track.tracks[0].pianoroll

        chroma_bass = to_chroma(bass_pianoroll)
        chroma_melody = to_chroma(melody_pianoroll)

        harmonicity[array_index] = get_harmonicity(chroma_melody, chroma_bass, 50,
                                                   tonal_matrix = tonal_matrix)

        bass_metrics[array_index,:] = get_metrics(bass_pianoroll)
        melody_metrics[array_index, :] = get_metrics(melody_pianoroll)

    return harmonicity

def get_ground_truth_metrics(dir="./data_processed/midis_tracks=Bass-Piano"):
    """
    Get harmonicity metric for a ground-truth dataset. Assumes songs have distinct
    bass and piano tracks that are recognizable by pypianoroll's Multitrack object.
    """

    bass_metrics = np.zeros((100, 3))
    melody_metrics = np.zeros((100, 3))

    harmonicity = np.zeros(100)

    tonal_matrix = get_tonal_matrix()

    array_index = 0
    for track in glob.glob(os.path.join(dir,"*.mid")):

        try:
            comb_track = Multitrack(track)
            first_track_name = comb_track.tracks[0].name
            second_track_name = comb_track.tracks[1].name
        except:
            continue

        if first_track_name == "Piano" and second_track_name == "Bass":
            melody_pianoroll = comb_track.tracks[0].pianoroll
            bass_pianoroll = comb_track.tracks[1].pianoroll
        elif first_track_name == "Bass" and second_track_name == "Piano":
            melody_pianoroll = comb_track.tracks[1].pianoroll
            bass_pianoroll = comb_track.tracks[0].pianoroll
        else:
            print(first_track_name, second_track_name)
            continue

        chroma_bass = to_chroma(bass_pianoroll)
        chroma_melody = to_chroma(melody_pianoroll)

        harmonicity[array_index] = get_harmonicity(chroma_melody, chroma_bass, 50,
                                                   tonal_matrix = tonal_matrix)

        bass_metrics[array_index,:] = get_metrics(bass_pianoroll)
        melody_metrics[array_index, :] = get_metrics(melody_pianoroll)
    
        array_index += 1
        
        if array_index >= 100:
            break

    return harmonicity[:array_index]

def get_metrics(pianoroll):
    """
    Helper function to compute some of the other evaluation metrics such
    as number of pitches used, number of pitch classes used, and the rate
    of polyphonicity.
    """
    n_pitches = metrics.n_pitches_used(pianoroll)
    n_classes = metrics.n_pitche_classes_used(pianoroll)
    poly_rate = metrics.polyphonic_rate(pianoroll, threshold=1)

    return np.array([n_pitches, n_classes, poly_rate])


if __name__ == "__main__":

    ground_truth_metrics = get_ground_truth_metrics()
    unconditional_metrics = get_dataset_metrics(base_term="unconditional")
    conditional_metrics = get_dataset_metrics(base_term="conditional")

    # Plot tonal distance distributions
    import matplotlib.pyplot as plt
    plt.hist([unconditional_metrics, conditional_metrics, ground_truth_metrics])
    plt.xlabel("Tonal Distance")
    plt.ylabel("Count")
    
    plt.show()

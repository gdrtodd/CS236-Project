import music21 as m21
import numpy as np
from fractions import Fraction
from collections import Counter


def get_vocab(upper_limit=8):
    '''
    Returns the list of all tokens in the vocabulary
    '''
    all_timings = []
    for i in np.arange(0, upper_limit, 0.25):
        all_timings.append(Fraction(i))

    for i in np.arange(0, upper_limit*3):
        if i%3 != 0:
            all_timings.append(Fraction(i, 3))

    vocab = []

    for note in range(128):
        vocab.append("on_{}".format(note))

    for timing in all_timings:
        vocab.append("dur_{}".format(str(timing)))
        vocab.append("adv_{}".format(str(timing)))

    vocab.append("off_note")
    vocab.append("<start>")
    vocab.append("<end>")

    return vocab

def get_closest_timing(timing, upper_limit=8):
    '''
    Returns the closest allowed timing to a provided value,
    as a Fraction object
    '''
    all_timings = []
    for i in np.arange(0, upper_limit, 0.25):
        all_timings.append(i)

    for i in np.arange(0, upper_limit*3):
        if i%3 != 0:
            all_timings.append(Fraction(i, 3))

    closest_idx = (np.abs(np.asarray(all_timings) - timing)).argmin()

    return Fraction(all_timings[closest_idx])

def encode(stream, highest_note_only=False, trim_silences=True):
    encoding = ['<start>']

    # The offset value of the current note / rest / chord
    cur_offset = 0.0

    # The offset value of the first event in the stream. If trim_silences
    # is set to True, then this will reflect the offset of the first *sound*
    # in the stream. Its value is subtracted from other offset values when
    # encoding
    start_offset = 0.0

    flattened = stream.flat

    if trim_silences:
        for idx, element in enumerate(flattened):
            if type(element) == m21.note.Note or type(element) == m21.chord.Chord:
                start_offset = element.offset
                break

        flattened = flattened[idx:]

    for element in flattened:
        if element.offset - start_offset > cur_offset:
            delta = get_closest_timing(element.offset - start_offset - cur_offset)
            encoding.append("adv_{}".format(delta))

            cur_offset = element.offset - start_offset

        duration = get_closest_timing(element.duration.quarterLength)

        all_durs.append(duration)
        if isinstance(element, m21.note.Note):
            encoding.append("on_{}".format(element.pitch.midi))

        elif isinstance(element, m21.chord.Chord):
            if highest_note_only:
                pitch = max(element.pitches)
                encoding.append("on_{}".format(pitch.midi))

            else:
                for pitch in element.pitches:
                    encoding.append("on_{}".format(pitch.midi))

        elif isinstance(element, m21.note.Rest):
            encoding.append("off_note")

        encoding.append("dur_{}".format(duration))

    encoding.append("<end>")

    return encoding

def decode(encoding):
    # Trim off the <start> and <end> tokens
    encoding = encoding[1:-1]

    stream = m21.stream.Stream()

    cur_offset = 0.0
    cur_pitches = []

    for event in encoding:
        name, extent = event.split('_')

        if name == 'on':
            cur_pitches.append(int(extent))
        elif name == 'off':
            cur_pitches.append('r')
        elif name == 'dur':
            quarter_duration = Fraction(extent)
            duration = m21.duration.Duration(quarter_duration)

            if len(cur_pitches) == 1:
                pitch = cur_pitches[0]
                if pitch == 'r':
                    note = m21.note.Rest(duration=duration)
                else:
                    note = m21.note.Note(pitch, duration=duration)

                stream.insert(cur_offset, note)

            elif len(cur_pitches) > 1:
                stream.insert(cur_offset, m21.chord.Chord(cur_pitches, duration=duration))

            cur_pitches = []

        elif name == 'adv':
            extent = Fraction(extent)
            cur_offset += extent

    return stream


if __name__ == '__main__':
    test_dir = './data_processed/midis_tracks=Piano/TRAACQE12903CC706C-piano.mid'
    stream = m21.converter.parse(test_dir)

    enc = encode(stream)
    dec = decode(enc)

    dec.show('midi')

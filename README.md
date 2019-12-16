# Learning to Groove: Conditional Melody Generation from Authentic Basslines

This code was developed for a final project for Stanford's CS 236 - Deep Generative
Models course (Autumn 2019).




### Authors

Graham Todd\
Collin Schlager\
Natalie Cygan


## Abstract

We seek to examine whether conditioning a melody-generating model on a richer
encoding of a bassline leads to better melody prediction and / or more coherent-sounding
music. We present an artificial “jam session” with an architecture that uses two generative
models: a bassline model that is trained first and provides rich encodings, and a melody
model that conditions generation upon those rich encodings. We also present a
novel encoding scheme for representing polyphonic music.

## Model Architecture and Encoding

<img src="https://github.com/gdrtodd/CS236-Project/blob/master/figures/model_diagram.png?raw=true" alt="model_diagram" width="50%"/>

LSTM architectures for the melody and bass-track models during training. The bass-track
model (lower rectangle) is fed a bass track in the form of encoded MIDI tokens. It generates a
sequence of artificially created bass-track tokens. These bass-track tokens are grouped together by
measure and fed into the melody model (upper rectangle), alongside a melody track from the same
song.

<img src="https://github.com/gdrtodd/CS236-Project/blob/master/figures/midi_encoding.png?raw=true" alt="encoding" width="70%"/>

An example MIDI track with its corresponding tuple encodings. Each note is assigned a
(pitch, duration, advance) value. Polyphony is achieved by having the advance value less than
the duration value. Notice the teal and yellow notes start simultaneously due to the teal note having
an advance value of zero.


## Listening to Samples

<img src="https://raw.githubusercontent.com/gdrtodd/CS236-Project/master/figures/conditional_cross_track_midi.png" alt="sample_midi" width="50%"/>

Check out the `./generated_samples/examples` directory for a small collection of samples.

## Generating Samples from Pre-trained Models

We have included some pre-trained models in the `./logs` directory. You can use these models
to sample new music generations for yourself!

The following command will generate 5 new samples and save them to ./generated_samples.
It will save the combined file (bass + melody) as well as the bass and melody as single
tracks.

`python3 sample_conditional_model.py --num_samples 5`

To compare our conditional model with a baseline unconditional model, you can generate
your own unconditioned samples using

`python3 sample_unconditional_model.py --num_samples 5`

## Training the Model

You can traing your own models using the following commands. Note that you will need
to unpack the dataset files (stored as .zip files on git) before beginning.

Train the bass-track model:

`python3 run_unconditioned_lstm.py --tracks Bass --num_epochs 10`

Before training the conditioned melody model, you will need to create a measure encoding file.
This file provides the bass-track hidden states for use by the conditional melody model during
training. You can generate this measure encoding file with

`python3 generate_measure_encodings.py --logdir <log_directory_of_bass_model> --tracks Bass`

where `<log_directory_of_bass_model>` should be filled in with the log directory that the
the bass track model you trained above is stored (it will likely be in `./logs`).

With the measure encoding file created, we can now train the conditioned model. Note: this
measure encoding file should be a `measure_encoding.pkl` file saved to the logdir provided.

Now, train the conditioned melody model using

`python3 run_conditioned_lstm.py --tracks Piano --measure_enc_dir <log_directory_of_bass_model>`

That's it! Take a look within these files for a full list of user-provided parameters.

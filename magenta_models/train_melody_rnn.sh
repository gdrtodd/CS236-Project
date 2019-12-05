melody_rnn_create_dataset \
--config='attention_rnn' \
--input=../data_processed/piano_notesequences.tfrecord \
--output_dir=logs/piano_melody_rnn/sequence_examples \
--eval_ratio=0.10

melody_rnn_train \
--config=attention_rnn \
--run_dir=logs/piano_melody_rnn/logdir/run1 \
--sequence_example_file=logs/piano_melody_rnn/sequence_examples/training_melodies.tfrecord \
--hparams="batch_size=64,rnn_layer_sizes=[64,64]" \
--num_training_steps=20000
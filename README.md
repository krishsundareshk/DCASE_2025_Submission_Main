# DCASE_2025_Submission_Main
A unified pipeline pools RGB spectrogram patches from all machines (with optional per-sample metadata), encodes them via a ResNet-34 backbone into 128-D vectors, and learns contrastively with NT-Xent loss. An attribute-conditioned attention layer weights patches, while ReduceLROnPlateau and early stopping stabilize training.

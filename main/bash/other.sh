cd ..

python training_real.py -ln arrhythmia -rt ../data/arrhythmia.mat -la 1 -nt arrhythmia_mlp_one_class -op one_class
python training_real.py -ln cardio -rt ../data/cardio.mat -la 1 -nt cardio_mlp_one_class -op one_class
python training_real.py -ln thyroid -rt ../data/thyroid.mat -la 1 -nt thyroid_mlp_one_class -op one_class
python training_real.py -ln shuttle -rt ../data/shuttle.mat -la 1 -nt shuttle_mlp_one_class -op one_class

python training_real.py -ln arrhythmia -rt ../data/arrhythmia.mat -nt arrhythmia_mlp_one_class -op one_class_unsupervised
python training_real.py -ln cardio -rt ../data/cardio.mat -nt cardio_mlp_one_class -op one_class_unsupervised
python training_real.py -ln thyroid -rt ../data/thyroid.mat -nt thyroid_mlp_one_class -op one_class_unsupervised
python training_real.py -ln shuttle -rt ../data/shuttle.mat -nt shuttle_mlp_one_class -op one_class_unsupervised

python training_real.py -ln arrhythmia -rt ../data/arrhythmia.mat -la 1 -nt arrhythmia_mlp_rec -op rec
python training_real.py -ln cardio -rt ../data/cardio.mat -la 1 -nt cardio_mlp_rec -op rec
python training_real.py -ln thyroid -rt ../data/thyroid.mat -la 1 -nt thyroid_mlp_rec -op rec
python training_real.py -ln shuttle -rt ../data/shuttle.mat -la 1 -nt shuttle_mlp_rec -op rec

python training_real.py -ln arrhythmia -rt ../data/arrhythmia.mat -nt arrhythmia_mlp_rec -op rec_unsupervised
python training_real.py -ln cardio -rt ../data/cardio.mat -nt cardio_mlp_rec -op rec_unsupervised
python training_real.py -ln thyroid -rt ../data/thyroid.mat -nt thyroid_mlp_rec -op rec_unsupervised
python training_real.py -ln shuttle -rt ../data/shuttle.mat -nt shuttle_mlp_rec -op rec_unsupervised

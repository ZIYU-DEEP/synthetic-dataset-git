cd ..

python training_real.py -ln satimage -rt ../data/satimage-2.mat -la 1 -nt satimage_mlp -op one_class
python training_real.py -ln satimage -rt ../data/satimage-2.mat -la 1 -nt satimage_mlp -op rec
python training_real.py -ln satimage -rt ../data/satimage-2.mat -nt satimage_mlp -op one_class_unsupervised
python training_real.py -ln satimage -rt ../data/satimage-2.mat -nt satimage_mlp -op rec_unsupervised

python training_real.py -ln satellite -rt ../data/satellite.mat -la 1 -nt satellite_mlp -op one_class
python training_real.py -ln satellite -rt ../data/satellite.mat -la 1 -nt satellite_mlp -op rec
python training_real.py -ln satellite -rt ../data/satellite.mat -nt satellite_mlp -op one_class_unsupervised
python training_real.py -ln satellite -rt ../data/satellite.mat -nt satellite_mlp -op rec_unsupervised

# wpynmt

1. update on 2018.2.7

	a. update the file structures, different models corresponding to different searchers;
	b. fix some other small bugs;
	c. add some tricks:
			such as copy trg vocab weight;
			fit segmentation for source sentence;
			add layer norm for gru.py;
			with bpe support, but need to preprocess by open-source bpe tools;
			add rn model and sru model;
	d. fix the force-decoding alignment generation for AER calculation when translating a file


# translate

python wtrans.py -m model_file -i 900

# evaluate alignment

score-alignments.py -d path/900 -s zh -t en -g wa -i force_decoding_alignment


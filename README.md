#  unknown_number_source_separation
Underwater single-channel acoustic source separation with unknown numbers using autoencoder nerual networks. Using keras-gpu 2.2.4 with tensorflow-gpu 1.12.0 backend.  
#  How to cite this work
This is the official code of the blow article.
Please cite this work in your publications as :  
<pre>
@misc{sun2022source,
      title={Source Separation of Unknown Numbers of Single-Channel Underwater Acoustic Signals Based on Autoencoders}, 
      author={Qinggang Sun and Kejun Wang},
      year={2022},
      eprint={2207.11749},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
} 
</pre>
#  How to use
<pre>
1. Download and organize data files as "data_dir_tree.md".
    The official website of the shipsEar database is https://atlanttic.uvigo.es/underwaternoise/ .
    Users may contact the author David Santos Domínguez through email dsantos@gts.uvigo.es to get the database.
2. Install dependent packages in the "requirements.txt" file.
3. Run
	prepare_data_shipsear_recognition_mix_s0tos3.py
	recognition_mix_shipsear_s0tos3_preprocess.py
	prepare_data_shipsear_separation_mix_s0tos3.py
	separation_mix_shipsear_s0tos3_preprocess.py
   to preprocess datas.
4. Experiments:
	(1) Unknown numbers of source separation.
		train_separation_multiple_autoencoder.py
	(2) Algorithm 1.
		train_separation_one_autoencoder.py
	(3) Algorithm 2.
		train_single_source_autoencoder_ns.py
		train_single_source_autoencoder_ns_search_encoded.py
	(4) Algorithm 3.
		train_single_source_autoencoder_ns.py
		train_separation_one_autoencoder_freeze_decoder.py
</pre>
#  Reference
Please cite the original work as :  
[Generative sourceseparation with GANs](https://github.com/ycemsubakan/sourceseparation_misc#generative-sourceseparation-with-gans)  
[Two-Step Sound Source Separation: Training on Learned Latent Targets](https://github.com/etzinis/two_step_mask_learning#two-step-sound-source-separation-training-on-learned-latent-targets)  
[Dual-Path RNN for Single-Channel Speech Separation](https://github.com/sp-uhh/dual-path-rnn#dual-path-rnn-for-single-channel-speech-separation-in-keras-tensorflow2)  
[Sound Separation](https://github.com/google-research/sound-separation)  
[bsseval](https://github.com/sigsep/bsseval)  

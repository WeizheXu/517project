#! /bin/bash
export PYTHONPATH="$PYTHONPATH:/home/xuweizhe/517_project/AC-MIMLLN-master"

# Default models:
# AC-MIMLLN: REST14
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN REST14 w/o mil
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil False --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN w/o mil MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil False --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN-Affine REST14
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN-Affine MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer fc --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN-Bert REST14
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --bert_file_path /home/xuweizhe/517_project/bert-base-uncased.tar.gz --bert_vocab_file_path /home/xuweizhe/517_project/bert-base-uncased-vocab.txt --data_type mil-bert --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN-Bert MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --bert_file_path /home/xuweizhe/517_project/bert-base-uncased.tar.gz --bert_vocab_file_path /home/xuweizhe/517_project/bert-base-uncased-vocab.txt --data_type mil-bert --current_dataset MAMSACSA --mil True --bert True --pair True --joint_type warmup --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 16 --train True --evaluate True --evaluation_on_instance_level False


# Multitask learning experiments
# AC-MIMLLN: REST14, joint + single
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-single --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMACSA, joint + single
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-single --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN: REST14, joint + multi = default

# AC-MIMLLN: REST14, pipeline + multi
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type pipeline --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMACSA, pipeline + multi
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type pipeline --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN: REST14, pipeline + single
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type pipeline --acd_sc_mode multi-single --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMACSA, pipeline + single
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type pipeline --acd_sc_mode multi-single --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 3 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False




# Layer 1 models:
# AC-MIMLLN: REST14
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 1 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 1 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False

#Layer 2 models:
# AC-MIMLLN: REST14
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset SemEval-2014-Task-4-REST-DevSplits --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 2 --batch_size 32 --train True --evaluate True --evaluation_on_instance_level False

# AC-MIMLLN MAMSACSA
time python nlp_tasks/absa/aspect_category_detection_and_sentiment_classification/acd_and_sc_bootstrap_pytorch_mil.py --embedding_filepath /home/xuweizhe/517_project/glove.840B.300d.txt --data_type mil --current_dataset MAMSACSA --mil True --bert False --pair False --joint_type joint --acd_sc_mode multi-multi --lstm_or_fc_after_embedding_layer lstm --lstm_layer_num_in_lstm 2 --batch_size 64 --train True --evaluate True --evaluation_on_instance_level False



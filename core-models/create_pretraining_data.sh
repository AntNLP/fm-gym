input_file='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/segmented_wikipedia_en_100k.txt'
output_file='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/training_data_wikipedia_en_100k.jsonl'
vocab_file='./vocab.txt'

python create_pretraining_data.py \
    --input_file $input_file \
    --output_file $output_file \
    --vocab_file $vocab_file \
    --do_lower_case True \
    --do_whole_word_mask False \
    --max_seq_length 512 \
    --max_predictions_per_seq 77 \
    --random_seed 42 \
    --dupe_factor 1 \
    --short_seq_prob 0.1

# The max_predictions_per_seq is the maximum number of masked LM predictions per sequence. You should set this to around max_seq_length * masked_lm_prob (the script doesn't do that automatically because the exact value needs to be passed to both scripts). 

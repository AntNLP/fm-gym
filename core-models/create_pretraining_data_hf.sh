segmented_file_path='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/segmented_wikipedia_en_100k.txt'
output_file_path='/mnt/data1/public/corpus/Bert_Pretrain/Raw_Wikipedia_EN/training_data_wikipedia_en_100k.jsonl'

python hf4mlm.py \
	--segmented_file_path $segmented_file_path
	--output_file_path $output_file_path 
	--model_name bert-base-uncased \
	--block_size 512 \
	--batch_size 32

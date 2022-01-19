from pydoc import describe
import spacy
import argparse
from tqdm import tqdm
from datasets import load_dataset


def main(args):
    dataset = load_dataset("wikipedia", "20200501.en", cache_dir="/mnt/data1/public/corpus/Bert_Pretrain/Wikipedia_EN/")

    raw_dataset = dataset["train"]
    output = open(args.output_file, 'w')

    nlp = spacy.load('en_core_web_sm')
    for text in tqdm(raw_dataset[:args.document_num]["text"]):
        # filte \n \t \'
        text = text.replace('\n', '')
        text = text.replace('\t', '')
        text = text.replace('\'', "'")
            
        sentences = [sent.text for sent in nlp(text).sents]
        for sent in sentences:
            output.write(sent + '\n')
        output.write('\n')
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, required=True, help='the path of output file')
    parser.add_argument("--document_num", type=int, default=100, help='the number of processing document (from front)')
    args = parser.parse_args()
    main(args)

import os
import json
from operator import itemgetter
import argparse
import nltk
from collections import Counter

def process_split(args, split_name):
    """extract answers that obtain highest votes."""

    # Load question
    que_fname = '{}/Questions/MultipleChoice_mscoco_{}_questions.json'
    que_fname = que_fname.format(args.vqa_path, split_name)
    print('[Loading] {}'.format(que_fname))
    with open(que_fname) as f:
        questions = json.load(f)['questions']

    # Tokenize question
    print('[Info] Tokenizing')
    for q in questions:
        q['question'] = ' '.join(nltk.word_tokenize(q['question'].lower()))

    # Extract answer
    ann_fname = '{}/Annotations/mscoco_{}_annotations.json'
    ann_fname = ann_fname.format(args.vqa_path, split_name)
    if os.path.exists(ann_fname):
        # Index question by `question_id`
        questions = {q['question_id']:q for q in questions}

        print('[Loading] {}'.format(ann_fname))
        with open(ann_fname) as f:
            annotations = json.load(f)['annotations']
        for ann in annotations:
            counter = Counter(map(itemgetter('answer'), ann['answers']))
            top_answer = counter.most_common(1)[0][0]
            qid = ann.get('question_id')
            questions[qid]['answer'] = top_answer
        questions = [v for k,v in questions.items()]

    out_fname = '{}/{}.json'.format(args.out, split_name)
    print('[Saving] {}'.format(out_fname))
    with open(out_fname, 'w') as f:
        json.dump(questions, f)

def main(args):
    for split_name in ('train2014', 'val2014', 'test-dev2015', 'test2015'):
        print('[Info] processing %s...' % (split_name,))
        process_split(args, split_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract samples from VQA dataset.')

    parser.add_argument('--vqa_path', default='data/vqa', help='vqa tools path')
    parser.add_argument('--out', default='data', help='output directory')

    args = parser.parse_args()
    main(args)

import json
from operator import itemgetter
from collections import Counter
import pandas as pd
import numpy as np
import argparse
import os.path


def get_qtype(qtype, question):
    choice = 'none of the above'
    for t in qtype:
        question = ' '.join(question.split()).lower() # avoid duplicate blank
        if question.startswith(t+' ') or question.startswith(t+'?'):
            choice = t
            break
    return choice


def compute_over_answer(qtype, answer, sample):
    table = pd.DataFrame(
            np.zeros((len(qtype)+1, len(answer)+1), dtype='int32'),
            index=qtype+['none of the above'],
            columns=answer+['**other**'])
    for s in sample:
        a = s['answer']
        qt = get_qtype(qtype, s['question'])
        if a in answer:
            table.loc[qt, s['answer']] += 1
        else:
            table.loc[qt, '**other**'] += 1
    return table


def compute_total(qtype, sample):
    table = pd.DataFrame(
            np.zeros((len(qtype)+1, 1), dtype='int32'),
            index=qtype+['none of the above'],
            columns=['total'])
    for s in sample:
        qt = get_qtype(qtype, s['question'])
        table.loc[qt, 'total'] += 1
    return table


def main(args):
    split_names = ('train2014', 'val2014', 'test-dev2015', 'test2015')
    # load data
    data = {}
    for split in split_names:
        file_path = '{}/{}.json'.format(args.data_dir, split)
        print('[Loading] {}'.format(file_path))
        with open(file_path) as f:
            data[split] = json.load(f)

    # get top answer
    answer = map(itemgetter('answer'), data['train2014'])
    answer_count = Counter(answer).most_common(args.topk)
    answer = map(itemgetter(0), answer_count)
    print('Pick top {} answers'.format(len(answer)))

    # load question type
    ques_type_file = '{}/QuestionTypes/mscoco_question_types.txt'.format(args.vqa_dir)
    print('[Loading] {}'.format(ques_type_file))
    with open(ques_type_file) as f:
        qtype = f.read().strip().split('\n')
    qtype.sort(reverse=True)

    # compute distribution
    writer = pd.ExcelWriter(args.out, engine='xlsxwriter')
    for split in split_names:
        if 'answer' in data[split][0]:
            table = compute_over_answer(qtype, answer, data[split])
        else:
            table = compute_total(qtype, data[split])
        table.to_excel(writer, sheet_name=split)
    print('[Saving] {}'.format(args.out))
    writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the distribution of samples over question type and top answers.')
    parser.add_argument('--data_dir', default='data', help='data directory')
    parser.add_argument('--vqa_dir', default='data/vqa', help='vqa tools directory')
    parser.add_argument('--topk', default=1000, help='number of top answer will be accounted')
    parser.add_argument('--out', default='sample_dist.xlsx', help='path of output file')
    args = parser.parse_args()
    main(args)

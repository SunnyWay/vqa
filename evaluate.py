# coding: utf-8

from __future__ import print_function

import sys
dataDir = 'data/vqa'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
sys.path.insert(0, '%s/PythonEvaluationTools' % (dataDir,))
from vqa import VQA
from vqaEvaluation.vqaEval import VQAEval
import json
import random
import os
import argparse

def main(params):
    # set up file names and paths
    taskType    =params['task']
    dataType    ='mscoco'  # 'mscoco' for real and 'abstract_v002' for abstract
    dataSubType = 'val2014'
    annFile     ='%s/Annotations/%s_%s_annotations.json'%(dataDir, dataType, dataSubType)
    quesFile    ='%s/Questions/%s_%s_%s_questions.json'%(dataDir, taskType, dataType, dataSubType)

    resultPath = params['res_file'].rsplit('/', 1)[0]
    resultPath = '.' if resultPath == params['res_file'] else resultPath
    resultType = params['res_file'].rsplit('_', 1)[0].rsplit('/', 1)[-1]
    fileTypes   = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

    # An example result json file has been provided in './Results' folder.  

    resFile = params['res_file']
    [accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = \
        ['%s/%s_%s_%s_%s_%s.json'%(resultPath, taskType, dataType, dataSubType, resultType, fileType) \
            for fileType in fileTypes]  

    # create vqa object and vqaRes object
    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(resFile, quesFile)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate() 

    # print accuracies
    #print "\n"
    print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), file=sys.stderr)
    #print "Per Question Type Accuracy is the following:"
    #for quesType in vqaEval.accuracy['perQuestionType']:
    #        print "%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType])
    #print "\n"
    #print "Per Answer Type Accuracy is the following:"
    #for ansType in vqaEval.accuracy['perAnswerType']:
    #        print "%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType])
    #print "\n"

    # save evaluation results to ./Results folder
    print(accuracyFile)
    json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
    json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
    json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
    json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate results on VQA.')
    parser.add_argument('--res_file', required=True, help='json file contains pairs of question_id and answer')
    parser.add_argument('--task', required=True, help='MultipleChoice or OpenEnded')

    args = parser.parse_args()
    params = vars(args)
    main(params)

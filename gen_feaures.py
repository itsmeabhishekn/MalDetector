
# About: Malware detection using deep autoencoders
# Author: walid.daboubi@gmail.com
# Version: 1

import os
import sys

DATA_DIR = '/home/goku/lf/project/malware-detection-with-deep-learning-autoencoder/wannaCry_data/'
DEST_FILE = '/home/goku/lf/project/malware-detection-with-deep-learning-autoencoder/wannaCry_data/wannaCry.csv'

"""
DATA_DIR = './data/train/'
DEST_FILE = 'data_all.csv'

DATA_DIR = './benign_dataset/'
DEST_FILE = 'data_benign.csv'
"""

ALL_ASM_OPS = ['AAA','AAD','AAM','AAS','ADC','ADD','AND','CALL',
           'CLD','CLI','CMC','CMP','CMPSB','CMPSW','CWD','STD'
           'CBW','CLC','DAA','DAS','DEC','DIV','ESC','HLT',
           'IDIV','IMUL','IN','INC','INT','INTO','IRET','JA',
           'JAE','JB','JBE','JC','JE','JG','JGE','JL','JLE',
           'JNA','JNAE','JNB','JNBE','JNC','JNE','JNG','JNGE',
           'JNL','JNLE','JNO','JNP','JNS','JO','JP','JPE','JPO',
           'JS','JZ','JCXZ','JMP','LAHF','LDS','LEA','LES','LOCK',
           'LODSB','LODSW','LOOP','LOOPNE','LOOPZ','MOV','MOVSB',
           'MOVSW','MUL','NEG','NOP','NOT','OR','OUT','POP','POPF',
           'PUSH','PUSHF','RCL','RCR','REP','REPNE','REPNZ','REPZ',
           'RET','RETN','RETF','ROL','ROR','SAHF','SAL','SAR','SBB',
           'SCASB','SCASW','SHL','SHR','STOSB','STOSW','SUB','TEST',
           'WAIT','XCHG','XLAT','XOR','JNZ','LOOPNZ','REPE','STC','STI']

ASM_OPS = ['MOV','PUSH','CALL','CMP','ADD','POP','LEA','TEST',
           'JE','XOR','JMP','JNE','RET','INC','SUB','AND','SHL','OR']

data = []

def get_features(asm_file):
    asm_file_dc = open(asm_file,  'r', encoding = "ISO-8859-1",)
    features = {}
    for op in ASM_OPS:
        features[op.lower()] = 0
    for line in asm_file_dc.readlines():
        line = line.replace('\t',' ')
        for op in ASM_OPS:
            if line.split(' ').count(op.lower()) > 0:
                features[op.lower()] += line.split(' ').count(op.lower())
    return features

result_data_file = open(DEST_FILE, 'w')
csv_initialized = False
for malware_file in os.listdir(DATA_DIR):
    result_data_file = open(DEST_FILE,'a')
    print(malware_file)
    current_features = get_features(DATA_DIR + malware_file)
    print(current_features)
    if csv_initialized == False:
        current_line = []
        for feature in current_features:
            current_line.append(str(feature))
        result_data_file.write(','.join(current_line) + '\n')
        csv_initialized = True
    current_line = []
    for feature in current_features:
        current_line.append(str(current_features[feature]))
    result_data_file.write(','.join(current_line) + '\n')
    result_data_file.close()

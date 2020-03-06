import re

filename = 'task2_en_training.tsv'

twt = ''

with open(filename, 'r') as f:
    twt = f.read()

re.sub(',', ' ', twt)

re.sub('\t', ',', twt)

with open('task2_en_training.csv', 'w+') as f:
    f.write(twt)

import glob2 as glob
import random


filelist = glob.glob('data/*')

with open('valid.txt', 'w') as file2:

    with open('train.txt', 'w') as file1:
        for files in filelist:
            if random.random() < 0.8:
                file1.writelines(files.split('/')[1] + '\n')
            else:
                file2.writelines(files.split('/')[1] + '\n')


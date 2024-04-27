# This file will be routinely updated

def save_dictionary(dir, dic):
    # save file in given directory as txt file, avoid json and pickle since keys are limited
    with open(dir,'w+') as f:
        f.write(str(dic))

def load_dictionary(dir):
    dic = ''
    with open(dir,'r') as f:
        for i in f.readlines():
            dic=i #string
    dic = eval(dic) # this is orignal dict with instace dict
    return dic

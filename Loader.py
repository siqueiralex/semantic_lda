from os import listdir
from datetime import datetime
from os.path import isdir, isfile, join

def date_from_filename(filename):
    digits =''.join([c for c in filename if c.isdigit()]) 
    year = 1900+int(digits[:2])
    day = int(digits[4:])
    month = int(digits[:-2][2:])
    return datetime(year=year,month=month,day=day)

def load_from_folder(folderpath, encoding = "ISO-8859-1"):
    corpus = {}
    folders = [f for f in listdir(folderpath) if isdir(folderpath+"/"+f) and f[0]!='.']
    articles=[]
    dates=[]
    
    for folder in folders:
        files = [f for f in listdir(folderpath+"/"+folder) if isfile(join(folderpath+"/"+folder, f)) and f[0]!='.']
    
        for file in files:
            file_articles = open(folderpath+"/"+folder+"/"+file, encoding=encoding).read().splitlines()
            dates.extend([date_from_filename(file)]*len(file_articles))
            articles.extend(file_articles)
    
    corpus['articles'] = articles
    corpus['dates'] = dates        
    return corpus


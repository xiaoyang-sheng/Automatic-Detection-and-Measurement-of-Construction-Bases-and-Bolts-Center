import os
import shutil
import stat
import time


def get_str_n(n: int):
    n = str(n)
    if(len(n) == 1):
        n = '0'+n
    return n


def forceDelete(fileName: str):
    if(not os.path.exists("./"+fileName)):
        return
    os.chmod("./"+fileName, stat.S_IRWXO+stat.S_IRWXG+stat.S_IRWXU)
    if(os.path.isdir("./"+fileName)):
        shutil.rmtree("./"+fileName)
        return
    os.remove("./"+fileName)


files = os.listdir(".")
files.sort()
new_files = []
for f in files:
    if(f[-4:] == '.jpg' or f[-4:] == '.png'):
        new_files.append(f)
n = len(new_files)
for i in range(0, n):
    old_name = './'+new_files[i]
    new_name = './'+get_str_n(i+1)
    if(old_name[-2] == 'p'):
        new_name = new_name+'.jpg'
    if(old_name[-2] == 'n'):
        new_name = new_name+'.png'
    shutil.copy(old_name, new_name)

for f in new_files:
    forceDelete(f)
forceDelete('./DS_Store')
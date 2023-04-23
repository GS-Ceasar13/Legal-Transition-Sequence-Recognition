from glob import *
from os import system
fileList = glob('1.pdf')
for f in fileList:
  system('pdftops -eps {0}'.format(f))



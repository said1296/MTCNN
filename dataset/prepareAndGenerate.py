from prepareDataset import prepareDataset
from generateTfrecord import generateTfrecord

prepareDataset(False,True,True)
generateTfrecord(pnetFlag=True)
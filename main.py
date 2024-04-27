from loftr import loftrGenerate
from findJpg import returnPair

print("please input path:")
path = input()

paired_files = returnPair(path)

for pair in paired_files:
    loftrGenerate(pair[0],pair[1])


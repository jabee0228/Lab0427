import os

def pair_jpg_files(directory):
    jpg_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jpg')]
    paired_files = []
    for i in range(len(jpg_files)):
        for j in range(len(jpg_files)):
            if i != j:
                paired_files.append((jpg_files[i], jpg_files[j]))
    return paired_files

def returnPair(path):

    paired_files = pair_jpg_files(path)

    for pair in paired_files:
        print(pair[0], pair[1])
    return paired_files

returnPair()
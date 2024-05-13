import os
import csv
import shutil

genres = [2, 5, 6]

with open("../../Datasets/wikiart_csv/genre_train.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        genre = int(row[1])
        if genre in genres:
            split = os.path.split(row[0])
            style = split[0]
            name = split[1]
            if style == "Early_Renaissance" or style == "High_Renaissance" or style == "Mannerism_Late_Renaissance":
                style = "Renaissance"
            if style == "Baroque" or style == "Renaissance" or style == "Impressionism":
                imagefrom = f"../../Datasets/wikiart/{split[0]}/{name}"
                imageto = f"../../Datasets/custom/WikiArtPortraitAndNudes/{style}/{name}"
                try:
                    shutil.copy(imagefrom, imageto)
                except:
                    pass
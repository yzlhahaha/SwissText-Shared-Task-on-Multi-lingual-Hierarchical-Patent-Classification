import csv
from Config import config
import html
import html2text
import sys
import csv
from tqdm import tqdm
import re

csv.field_size_limit(sys.maxsize)

def get_data(filename):
    ID = []
    data = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for i, line in enumerate(reader):
            ID.append(line[0])
            if filename == "Dev.csv" or filename == "Test.csv":
                continue
            if filename == "Train.csv":
                data[ID] = line[1:]
            else:
                data[ID] = line[1]
    return ID, data


def get_dataset(title_file, abstrast_file, description_file, id_file):
    IDs, labels = get_data(id_file)
    _, titles = get_data(title_file)
    _, abstracts = get_data(abstrast_file)
    _, descriptions = get_data(description_file)

    for ID in tqdm(IDs):
        with open("summary/"+id_file, "a+") as g:
            csv_write = csv.writer(g)
            data_row = [ID, titles[ID], abstracts[ID], descriptions[ID]]
            for label in labels[ID]:
                data_row.append(label)
            csv_write.writerow(data_row)


if __name__ == "__main__":
    train_file = "Train.csv"
    dev_file = "Dev.csv"
    test_file = "Dev.csv"
    title = "patent_TITLE.csv"
    abstract = "patent_ABSTR.csv"
    description = "patent_description.csv"
    get_dataset(title, abstract, description, train_file)

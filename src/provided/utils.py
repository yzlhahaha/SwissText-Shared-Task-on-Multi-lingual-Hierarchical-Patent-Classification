from sklearn_hierarchical_classification.constants import ROOT
import csv
import re
import html
import html2text

def extend_hierarchy(hierarchy, y_labs ):
    for samples_t in y_labs:
        if not isinstance(samples_t, list):            
            samples=[samples_t]
        else:
            samples=samples_t
        for lab in samples:
            par_1=lab[0]
            par_2=lab[:3]
            child=lab[:]
        
            if par_1 not in hierarchy[ROOT]:
                hierarchy[ROOT].append(par_1)
                print(lab, par_1, ROOT)
            if par_1 not in hierarchy:
                hierarchy[par_1]=[par_2]
            else:
                if par_2 not in hierarchy[par_1]:
                    hierarchy[par_1].append(par_2)
            if par_2 not in hierarchy:
                hierarchy[par_2]=[child]
            else:
                if child not in hierarchy[par_2]:  
                    hierarchy[par_2].append(child)
    return hierarchy
            


# hierarchy[parent]=[child1,...]
def build_hierarchy(issues):
    hierarchy={ROOT:[]}
    for i in issues:
        par_1=i[0]
        par_2=i[:3]
        child=i[:]
        
        if par_1 not in hierarchy[ROOT]:
            hierarchy[ROOT].append(par_1)
        if par_1 not in hierarchy:
            hierarchy[par_1]=[par_2]
        else:
            if par_2 not in hierarchy[par_1]:
                hierarchy[par_1].append(par_2)
        if par_2 not in hierarchy:
            hierarchy[par_2]=[child]
        else:
            hierarchy[par_2].append(child)
        
    return hierarchy


def preload():
    """
    extract and clean data from XML annotations
    """
    #read ids
    prepath= "Release_data/"
    train_ids=[]
    train_labels=[]
    with open(prepath+"Train.csv", 'r', encoding="utf-8") as f:

            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                labels =line[1]
                train_ids.append(ids)
                if "\t" in labels:
                    train_labels.append(labels.split("\t"))
                else:
                    train_labels.append(labels)
    dev_ids=[]
    dev_labels=[]
    with open(prepath+"Dev.csv", 'r', encoding="utf-8") as f:

            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                #labels =line[1]
                dev_ids.append(ids)
                #if "\t" in labels:
                #    dev_labels.append(labels.split("\t"))
                #else:
                #    dev_labels.append(labels)
    test_ids=[]
    test_labels=[]
    with open(prepath+"Test.csv", 'r', encoding="utf-8") as f:

            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                #labels =line[1]
                test_ids.append(ids)
                # if "\t" in labels:
                #     test_labels.append(labels.split("\t"))
                # else:
                #     test_labels.append(labels)
    dev_ids_g=[]
    dev_labels_g=[]
    with open("GT/Dev_GT.csv", 'r', encoding="utf-8") as f:

            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                labels =line[1:]
                dev_ids_g.append(ids)

                dev_labels_g.append(labels)
                
    test_ids_g=[]
    test_labels_g=[]
    with open("GT/Test_GT.csv", 'r', encoding="utf-8") as f:

            reader = csv.reader(f, delimiter="\t")
            for i, line in enumerate(reader):
                if i==0:
                    continue
                ids = line[0]
                labels =line[1]
                test_ids_g.append(ids)
                test_labels_g.append(labels)

    x_train={}
    x_dev= {}
    x_test={}
    TAG_RE = re.compile(r'<[^>]+>')
    files=[prepath+"patent_ABSTR.csv",prepath+"patent_TITLE.csv","data_full/patent_description.csv"]
    for f1,fname in enumerate(files):
         with open(fname, 'r', encoding="utf-8") as f:
             reader = csv.reader(f, delimiter="\t")
             for i, line in enumerate(reader):
                 if i==0:
                     continue
                 ids = line[0]
                 for x in [(x_train,train_ids),(x_test,test_ids),(x_dev,dev_ids)]:
                     if ids in x[1]:
                         if f1<2:
                             if ids in x[0]:
                                 x[0][ids]+="\n\n "+html2text.html2text(html.unescape(line[1]))
                             else:
                                 x[0][ids]=html2text.html2text(html.unescape(line[1]))
                         else:
                             if ids in x[0]:
                                 x[0][ids]+="\n\n "+TAG_RE.sub('', line[1])
                             else:
                                 x[0][ids]=TAG_RE.sub('', line[1])

    x_train = [ x_train[tk] for tk  in train_ids if tk!="Labels"]
    x_dev = [ x_dev[tk] for tk  in dev_ids  if tk!="Labels"]
    x_test = [ x_test[tk] for tk  in test_ids  if tk!="Labels"]
    return x_train, train_labels, x_dev, dev_labels, x_test, test_labels, train_ids, dev_ids, test_ids, dev_ids_g, dev_labels_g, test_ids_g, test_labels_g

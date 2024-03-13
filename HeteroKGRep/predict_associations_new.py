import argparse
import pickle
import torch
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


from .utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True) 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    args = {'embeddingf':args.embedding_file,
     'pairf':args.pair_file, 
     'seed':args.seed,
     'patience':args.patience,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.validation_ratio
    }
    return args

def get_drug_idx(drug, nodes):

    if drug in nodes:
        return nodes.index(drug)
    else:
        raise Exception(f"Drug {drug} not in nodes")

    return drug_idx

def get_dis_idx(dis, nodes):

    dis_idx = None

    if dis in nodes:  
        return nodes.index(dis)
    else:
        raise Exception(f"Disease {dis} not in nodes")

    return dis_idx

def load_embeddings(embedding_file):
    
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    embeddings = torch.tensor(embeddings) 
    embeddings = embeddings.detach()
    
    return embeddings

def get_nodes(pair_file):
    
    nodes = []
    
    with open(pair_file) as f:
        
        for line in f:
            try:
                fields = line.split("\t")
                drug, dis, label = fields
            except:
                print("Error parsing line:", line)
                continue
            nodes.append(drug)
            nodes.append(dis)

    return list(set(nodes))

def read_lines(pair_file):
    with open(pair_file) as f:
        lines = f.readlines()

    return lines

def split_dataset(pairf, embeddingf, validr, testr, seed):
    
    embeddings = load_embeddings(embeddingf)
    nodes = get_nodes(pairf)
    
    xs = []
    ys = []
    
    for line in read_lines(pairf):
        
        drug_idx = None
        dis_idx = None
        
        drug, dis, label = line.split("\t")
        
        try:
            drug_idx = get_drug_idx(drug, nodes)
        except Exception as e:  
            print(e)
            continue
            
        try:    
            dis_idx = get_dis_idx(dis, nodes)
        except Exception as e:
            print(e)  
            continue
            
        if drug_idx is not None and dis_idx is not None:
            drug_emb = np.array(embeddings[drug_idx])
            dis_emb = np.array(embeddings[dis_idx])
    
            xs.append(drug_emb - dis_emb)       
            ys.append(int(label))
    
    x_train, x_test, y_train, y_test = train_test_split(
    xs, ys, test_size=testr+validr, random_state=seed)
    
    return x_train, x_test, y_train, y_test
    

def return_scores(target_list, pred_list):
    
    metric_list = [
        accuracy_score,  
        roc_auc_score,  
        average_precision_score,
        f1_score
    ]  
    
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list,pred_list))
        else: # accuracy_score, f1_score
            scores.append(metric(target_list, pred_list.round()))
    return scores


def predict_associations(embedding_file, pair_file, model_file, 
                         seed:int=42, validr:float=0.1, testr:float=0.1):
    
    set_seed(seed)
    x_train, x_test, y_train, y_test = split_dataset(pair_file, embedding_file,  
                       validr, testr, seed)

    clf = xgb.XGBClassifier()
    
    clf.fit(x_train, y_train)

    y_pred = clf.predict_proba(x_test)[:,1]
    scores = return_scores(y_test, y_pred)
    print(scores)

    clf.save_model(model_file)

    return scores
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
import pickle


import networkx as nx

def read_graph(edgeList, weighted=True, directed=False, delimiter="\t"):
    G = nx.MultiDiGraph()
    
    with open(edgeList) as f:
        for line in f:
            values = line.split(delimiter)
            #print(values)
            if len(values) != 5:
                print(f"Invalid line format: {line}")
                continue
    
            source = values[0]
            target = values[1]
            relation = int(values[2])
            #print(source, target)
            try:
                similarity = float(values[3])
                label = int(values[4])
                #print(label)
            except ValueError:
                print(f"Error converting values: {line}")
                continue
                      
            node_attrs = {'similarity': similarity, 'label': relation} 

            edge_attrs = {'relation': label, 
                          'similarity': similarity,
                          'label': relation}

            # Ajout des noeuds
            G.add_node(source, attrs=node_attrs)
            G.add_node(target, attrs=node_attrs)
            #node = G.nodes[source]
            #print(node['attrs'])

            # Ajout de l'arête  
            G.add_edge(source, target, attrs=edge_attrs)
        
    if not directed:
        G = G.to_undirected()
    
    #nx.write_gpickle(G, "demo/graph.pkl")
    pickle.dump(G, open("demo/graph.pkl","wb"))
    return G

def get_node_features(graph):
    return np.array([data['attrs']['similarity'] for node, data in graph.nodes(data=True)])

def get_node_labels(graph):
    labels = [data['attrs']['label'] for node, data in graph.nodes(data=True)]
    classes = list(set(labels)) 
    label_map = {l: i for i, l in enumerate(classes)}
    return [label_map[l] for l in labels]

# Dictionnaire pour stocker les attributs
node_attrs = {}  

def set_node_features(graph, features):
    
    global node_attrs
    
    for node, feat in zip(graph.nodes(), features):
        
        # Mettre à jour le dictionnaire d'attributs
        node_attrs[node] = feat  

    # Alternatif pour mettre plusieurs features par noeud
    #node_attrs[node].update(feat)
    
    return graph

from imblearn.over_sampling import SMOTE

def apply_smote(graph):
    
    X = np.array(get_node_features(graph)).reshape(-1,1)
    #print(X)
    #y1 = get_node_labels(graph)
    #print(y1)
    y = np.array(get_node_labels(graph)).reshape(-1,1)
    #print(y)
    
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X = imputer.fit_transform(X)
    X[np.isnan(X)] = 0
    df = pd.DataFrame(X) 
    df = df.fillna(0)
    X = df.to_numpy()
    
    y[np.isnan(y)] = 0
    
    #print(X.shape) 
    #print(y.shape)
    #unique, counts = np.unique(y, return_counts=True)
    #print(unique)
    #print(counts)
    
    #print("Avant affichage message")
    #min_count = min(counts)
    #threshold = 6
    #if min_count < threshold:
    #    print("Classe trop petite")
    
    
    sm = SMOTE()
    
    X_res, y_res = sm.fit_resample(X, y)
    
    # Mise à jour des features dans le graphe d'origine
    augmented_graph = set_node_features(graph, X_res)
    print("Augmenté avec succès")
    return augmented_graph

def save_graph(graph, filename):
    
    #nx.write_edgelist(graph, filename, encoding='utf-8')
    #nx.write_gpickle(graph, "demo/augmented_graph.pkl")
    pickle.dump(graph, open(filename,"wb"))
    print("Fichier pickle créé")
    # ou pour specifier le format
    #nx.write_gpickle(graph, filename, encoding='utf-8')
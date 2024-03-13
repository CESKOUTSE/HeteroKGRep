import torch
import torch_geometric
import random
from torch.nn import Embedding
from .data_augmentation import get_node_labels

# Chargement du modèle
def load_model(path):
    return torch.load(path)

# Forward pass 
def forward(model, graphs):
    drug_emb, disease_emb = model.forward(graphs)
    return drug_emb, disease_emb

# Marche aléatoire
def random_walk(nodes):
    
    seq = [random.choice(nodes)]   
    
    while len(seq) < 50:
        
        cur_node = seq[-1]   

        next_node = random.choice(nodes)

        seq.append(next_node)

    return seq

# Génération des séquences  
#def generate_sequences(nodes, embedding_dim):
    
#    string_to_id = {}
    
#    for i, node in enumerate(nodes):
#        if type(node) is str:  
#            string_to_id[node] = i
            
#    emb_layer = Embedding(len(string_to_id)+1, embedding_dim)

#    sequences = []

#    for _ in range(10000):
        
#        seq = random_walk(nodes)

#        seq = [string_to_id[n] if type(n) is str else n for n in seq]

#        seq_tensor = emb_layer(torch.tensor(seq)).float()

#        sequences.append(seq_tensor)

#    return torch.stack(sequences, dim=0)

def generate_sequences(nodes, graph, embedding_dim):
    
    node_labels = get_node_labels(graph)
    
    string_to_id = {}

    for i, node in enumerate(nodes):
        if type(node) is str:  
            string_to_id[node] = i

    emb_layer = Embedding(len(string_to_id)+1, embedding_dim)

    sequences = []

    for _ in range(10000):
        
        seq = random_walk(nodes)
        
        node_ids = [string_to_id[n] if type(n) is str else n for n in seq]  

        features = emb_layer(torch.tensor(node_ids)).float()

        for i, node_id in enumerate(node_ids):
            
            if i < len(node_labels):
                label = node_labels[i]
            else:
                continue

#            sequence = {
#                'features': features[i],
#                'label': label
#            }
            sequence = (features[i], label)

            sequences.append(sequence)

    return sequences

# Extraction des embeddings
def get_embeddings(sequences):
    
    model = load_model("model.pt")
    
    embeddings = []
    
    for batch in sequences:
        
        emb = forward(model, batch)  
        
        embeddings.append(emb)
        
    return torch.cat(embeddings, dim=0)
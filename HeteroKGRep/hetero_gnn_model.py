import torch
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super().__init__()
        
        self.Linear1 = nn.Linear(input_size, hidden_size) 
        self.Linear2 = nn.Linear(hidden_size, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.Linear1(x)
        x = self.relu(x)
        x = self.Linear2(x)
        return x
    
class HeteroGNN(nn.Module):
    
    def __init__(self, drug_input_size, disease_input_size, hidden_size, output_size):
        
        super().__init__()
        
        self.drug_mlp = MLP(drug_input_size, hidden_size, hidden_size)
        self.disease_mlp = MLP(disease_input_size, hidden_size, hidden_size)
        
        self.output_layer = nn.Linear(hidden_size*2, output_size)
        
    def forward(self, features, label):
        
        drug_feats = features  
        disease_feats = features 
        
        #print(f"Shape drug feats: {drug_feats.shape}")
        #print(f"Shape disease feats: {disease_feats.shape}")
       
        drug_embeds = self.drug_mlp(drug_feats)
        #print(f"Shape drug embeds: {drug_embeds.shape}")
        disease_embeds = self.disease_mlp(disease_feats)
        #print(f"Shape disease embeds: {disease_embeds.shape}")
       
        concat_embeds = torch.concat([drug_embeds, disease_embeds], dim=0)
       
        return self.output_layer(concat_embeds) 
       
    def training_step(self, features, label):
        
        preds = self.forward(features, label)
        print(preds)
        
        label_tensor = torch.tensor([label], dtype=torch.float)
        print(label_tensor)
       
        #loss = nn.BCELoss()(preds, label_tensor)
        loss = nn.MSELoss()(preds, label_tensor)
        #print(type(loss))
        
        self.losses.append(loss)
       
        return loss
       
    def fit(self, data, optimizer, epochs):
        
        self.losses = []

        for epoch in range(1, epochs+1):
            
            for sequence in data:
                
                # Extraire les données du tuple  
                features = sequence[0]  
                label = sequence[1]

                # Calcul de la perte
                loss = self.training_step(features, label)   
      
                self.losses.append(loss)

            if epoch == epochs:
                print("Entraînement terminé")
                break

            return self.losses
    
    # Fonction pour récupérer les embeddings 
    def get_embeddings(model, data):
        
        embeddings = []
        
        for features, label in data:
            
            embeds = model.forward(features, label)
            embeddings.append((embeds, label))
        
        return embeddings

    # Fonction de prédiction
    def predict(model, data):
        
        preds = []
        
        embeddings = get_embeddings(model, data)
        
        for embeds, label in embeddings:
            
            out = model.output_layer(embeds)
            preds.append((out, label))

        return preds
# HeteroKGRep
The official code implementation for HeteroKGRep from our paper:
"Heterogeneous Knowledge Graph based Drug Repositioning" (Knowledge-Based Systems, 2024).

## Model description

HeteroKGRep is a drug repositioning framework that leverages heterogeneous biomedical knowledge graphs to discover novel drug–disease associations.
Unlike existing methods based on homogeneous data sources, HeteroKGRep integrates multimodal biomedical information (ontologies, pathways, literature, genetic data) to enrich representation learning and improve prediction accuracy.

## Pipeline Overview

#### Step 1 — SMOTE-based Graph Augmentation

Handles entity distribution imbalance in the heterogeneous knowledge graph.

Generates synthetic nodes and edges while preserving the graph structure.

#### Step 2 — Node Sequence Generation

Performs random walks to capture local and global structural context.

Learns entity embeddings from augmented graph structure.

#### Step 3 — HeteroGNN Representation Learning

Uses type-specific MLP modules for drugs, diseases, and other biomedical entities.

Extracts embeddings capturing semantic and structural relationships.

#### Step 4 — Association Prediction with XGBoost

Trains an XGBoost classifier on learned embeddings to predict novel drug–disease associations.

## Model Architecture
![HeteroKGRep Architecture](HeteroKGRep-model.png)

## Setup
Clone the repository and navigate to the project directory:
git clone https://github.com/CESKOUTSE/HeteroKGRep.git
cd HeteroKGRep

#### Requirements:

Python 3.8+

PyTorch >= 1.10

DGL (Deep Graph Library)

scikit-learn

imbalanced-learn (for SMOTE)

XGBoost

## Files

HeteroKGRep.ipynb → Main model training and evaluation

graph_augmentation.py → SMOTE-based graph augmentation

node_sequence.py → Random walk sequence generation

hetero_gnn.py → Heterogeneous GNN model

predict_xgboost.py → Association prediction

data/ → Biomedical knowledge graph data and processed embeddings

## Dataset

The knowledge graph integrates multiple biomedical data sources:

Ontologies (e.g., DrugBank, DO)

Biological pathways

Literature-based associations

Genetic and molecular interaction data

## Citation

If you use this work, please cite:

@article{CESKOUTSE2024112638,

      title = {HeteroKGRep: Heterogeneous Knowledge Graph based Drug Repositioning},

      journal = {Knowledge-Based Systems},

      volume = {305},

      pages = {112638},

      year = {2024},

      issn = {0950-7051},

      doi = {https://doi.org/10.1016/j.knosys.2024.112638},

      url = {https://www.sciencedirect.com/science/article/pii/S0950705124012723},

      author = {Ribot Fleury T. Ceskoutsé and Alain Bertrand Bomgni and David R. Gnimpieba Zanfack and Diing D.M. Agany and Bouetou Bouetou Thomas and Etienne Gnimpieba Zohim},

      keywords = {Deep learning, Drug repurposing, Biomedical heterogeneous graph},

      abstract = {The process of developing new drugs is both time-consuming and costly, often taking over a decade and billions of dollars to obtain regulatory approval. Additionally, the complexity of patent                     protection for novel compounds presents challenges for pharmaceutical innovation. Drug repositioning offers an alternative strategy to uncover new therapeutic uses for existing medicines.                         Previous repositioning models have been limited by their reliance on homogeneous data sources, failing to leverage the rich information available in heterogeneous biomedical knowledge graphs.                     We propose HeteroKGRep, a novel drug repositioning model that utilizes heterogeneous graphs to address these limitations. HeteroKGRep is a multi-step framework that first generates a similarity                   graph from hierarchical concept relations. It then applies SMOTE over-sampling to address class imbalance before generating node sequences using a heterogeneous graph neural network. Drug and                     disease embeddings are extracted from the network and used for prediction. We evaluated HeteroKGRep on a graph containing biomedical concepts and relations from ontologies, pathways and                           literature. It achieved state-of-the-art performance with 99% accuracy, 95% AUC ROC and 94% average precision on predicting repurposing opportunities. Compared to existing homogeneous                             approaches, HeteroKGRep leverages diverse knowledge sources to enrich representation learning. Based on heterogeneous graphs, HeteroKGRep can discover new drug-disease associations, leveraging                    de novo drug development. This work establishes a promising new paradigm for knowledge-guided drug repositioning using multimodal biomedical data.
                  }
                  
}

## Performance

HeteroKGRep achieves:

Accuracy: 99%

AUC ROC: 95%

Average Precision: 94%

These results outperform state-of-the-art homogeneous graph-based approaches in drug repositioning.

## License

This project is licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DiagnosticDataProcessor:
    def __init__(self, cosine_threshold=0.5, embedding_model='all-MiniLM-L6-v2'):
        """
        Process diagnostic data and create graph structure
        
        Args:
            cosine_threshold: Minimum cosine similarity to create edge between diagnostic actions
            embedding_model: SentenceTransformer model name for creating embeddings
        """
        self.cosine_threshold = cosine_threshold
        self.embedding_model_name = embedding_model
        self.encoder = None
        
        # Data storage
        self.original_df = None
        self.diagnostic_nodes = {}  # diagnostic_id -> node_index
        self.question_nodes = {}    # question_id -> node_index
        self.node_to_info = {}      # node_index -> (type, id, text)
        
    def load_data(self, df):
        """
        Load data from DataFrame
        
        Args:
            df: DataFrame with columns ['diagnostic_action_id', 'diagnostic_action_title', 
                                    'question_text', 'question_id']
        """
        required_columns = ['diagnostic_action_id', 'diagnostic_action_title', 'question_text', 'question_id']
        
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        self.original_df = df.copy()
        print(f"Loaded data: {len(df)} rows")
        print(f"Unique diagnostic actions: {df['diagnostic_action_id'].nunique()}")
        print(f"Unique questions: {df['question_id'].nunique()}")
        
        return self
    
    def create_embeddings(self, force_recreate=False):
        """
        Create sentence embeddings for diagnostic actions and questions
        """
        if self.encoder is None or force_recreate:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.encoder = SentenceTransformer(self.embedding_model_name)
        
        # Get unique diagnostic actions and questions
        diagnostic_texts = self.original_df.groupby('diagnostic_action_id')['diagnostic_action_title'].first()
        question_texts = self.original_df.drop_duplicates('question_id').set_index('question_id')['question_text']
        
        print("Creating embeddings...")
        print(f"- {len(diagnostic_texts)} diagnostic action titles")
        print(f"- {len(question_texts)} questions")
        
        # Create embeddings
        diagnostic_embeddings = self.encoder.encode(diagnostic_texts.tolist())
        question_embeddings = self.encoder.encode(question_texts.tolist())
        
        # Store embeddings with their IDs
        self.diagnostic_embeddings = dict(zip(diagnostic_texts.index, diagnostic_embeddings))
        self.question_embeddings = dict(zip(question_texts.index, question_embeddings))
        
        print("Embeddings created successfully!")
        return self
    
    def create_node_mapping(self):
        """
        Create mapping from diagnostic/question IDs to graph node indices
        """
        node_idx = 0
        
        # Create nodes for diagnostic actions (they come first)
        unique_diagnostics = self.original_df.groupby('diagnostic_action_id')['diagnostic_action_title'].first()
        
        for diag_id, diag_title in unique_diagnostics.items():
            self.diagnostic_nodes[diag_id] = node_idx
            self.node_to_info[node_idx] = ('diagnostic', diag_id, diag_title)
            node_idx += 1
        
        # Create nodes for questions (they come after diagnostics)
        unique_questions = self.original_df.drop_duplicates('question_id')[['question_id', 'question_text']]
        
        for _, row in unique_questions.iterrows():
            question_id = row['question_id']
            question_text = row['question_text']
            self.question_nodes[question_id] = node_idx
            self.node_to_info[node_idx] = ('question', question_id, question_text)
            node_idx += 1
        
        self.num_diagnostic = len(self.diagnostic_nodes)
        self.num_questions = len(self.question_nodes)
        self.total_nodes = node_idx
        
        print(f"Node mapping created:")
        print(f"- {self.num_diagnostic} diagnostic action nodes")
        print(f"- {self.num_questions} question nodes")
        print(f"- {self.total_nodes} total nodes")
        
        return self
    
    def create_node_features(self):
        """
        Create node feature matrix by combining all embeddings
        """
        # Prepare feature matrix
        embedding_dim = len(list(self.diagnostic_embeddings.values())[0])
        node_features = np.zeros((self.total_nodes, embedding_dim))
        
        # Add diagnostic embeddings (they come first in node order)
        for diag_id, embedding in self.diagnostic_embeddings.items():
            node_idx = self.diagnostic_nodes[diag_id]
            node_features[node_idx] = embedding
        
        # Add question embeddings (they come after diagnostics)
        for question_id, embedding in self.question_embeddings.items():
            node_idx = self.question_nodes[question_id]
            node_features[node_idx] = embedding
        
        self.node_features = node_features
        print(f"Node features created: shape {self.node_features.shape}")
        
        return self
    
    def create_edges(self):
        """
        Create edges for the graph:
        1. Between diagnostic actions (based on cosine similarity)
        2. Between diagnostic actions and their questions (parent-child relationship)
        """
        edge_list = []
        
        # 1. Create edges between diagnostic actions based on cosine similarity
        if len(self.diagnostic_embeddings) > 1:
            diag_ids = list(self.diagnostic_embeddings.keys())
            diag_embeddings = np.array([self.diagnostic_embeddings[diag_id] for diag_id in diag_ids])
            
            cos_sim_matrix = cosine_similarity(diag_embeddings)
            
            # Find similar diagnostic pairs
            similar_pairs = np.where(
                (cos_sim_matrix >= self.cosine_threshold) & 
                (cos_sim_matrix < 0.99)  # Avoid perfect self-similarity
            )
            
            diagnostic_edges = 0
            for i, j in zip(similar_pairs[0], similar_pairs[1]):
                if i != j:  # No self-loops
                    diag_i_id = diag_ids[i]
                    diag_j_id = diag_ids[j]
                    
                    node_i = self.diagnostic_nodes[diag_i_id]
                    node_j = self.diagnostic_nodes[diag_j_id]
                    
                    edge_list.append([node_i, node_j])
                    edge_list.append([node_j, node_i])  # Undirected
                    diagnostic_edges += 1
            
            print(f"Created {diagnostic_edges//2} diagnostic-diagnostic edge pairs (similarity-based)")
        
        # 2. Create edges between diagnostic actions and their questions
        question_edges = 0
        for _, row in self.original_df.iterrows():
            diag_id = row['diagnostic_action_id']
            question_id = row['question_id']
            
            diag_node = self.diagnostic_nodes[diag_id]
            question_node = self.question_nodes[question_id]
            
            # Create bidirectional edges
            edge_list.append([diag_node, question_node])
            edge_list.append([question_node, diag_node])
            question_edges += 1
        
        # Remove duplicate edges (in case same diagnostic-question pair appears multiple times)
        edge_list = list(set(tuple(edge) for edge in edge_list))
        edge_list = [list(edge) for edge in edge_list]
        
        print(f"Created {question_edges} diagnostic-question connections")
        
        # Convert to tensor
        if len(edge_list) == 0:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            self.edge_index = torch.LongTensor(edge_list).t().contiguous()
        
        print(f"Total edges in graph: {self.edge_index.shape[1]}")
        return self
    
    def create_graph_data(self):
        """
        Create PyTorch Geometric Data object
        """
        # Create node type labels (0 for diagnostic, 1 for questions)
        node_types = []
        for node_idx in range(self.total_nodes):
            node_type, _, _ = self.node_to_info[node_idx]
            node_types.append(0 if node_type == 'diagnostic' else 1)
        
        # Convert to tensors
        node_features = torch.FloatTensor(self.node_features)
        node_types = torch.LongTensor(node_types)
        
        # Create PyG data object
        self.graph_data = Data(
            x=node_features,
            edge_index=self.edge_index,
            node_types=node_types,
            num_diagnostic=self.num_diagnostic,
            num_questions=self.num_questions
        )
        
        print(f"Graph data created successfully!")
        print(f"- Nodes: {self.graph_data.x.shape[0]}")
        print(f"- Features per node: {self.graph_data.x.shape[1]}")
        print(f"- Edges: {self.graph_data.edge_index.shape[1]}")
        
        return self.graph_data
    
    def process_data(self, df):
        """
        One-stop method to process all data
        
        Args:
            df: DataFrame with diagnostic and question data
            
        Returns:
            PyTorch Geometric Data object
        """
        return (self.load_data(df)
                   .create_embeddings()
                   .create_node_mapping()
                   .create_node_features()
                   .create_edges()
                   .create_graph_data())
    
    def get_node_info(self, node_idx):
        """Get information about a specific node"""
        return self.node_to_info.get(node_idx, "Unknown node")

class DiagnosticGraphSAGE:
    def __init__(self, embedding_dim=128, hidden_dim=256, num_layers=2, device='cpu'):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device(device)
        self.model = None
        
    def build_model(self, input_dim):
        """Build the GraphSAGE model"""
        self.model = GraphSAGE(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.num_layers
        ).to(self.device)
        return self.model
    
    def train_unsupervised(self, data, num_epochs=100, lr=0.01, weight_decay=5e-4):
        """Train GraphSAGE in unsupervised manner"""        
        if self.model is None:
            self.build_model(data.x.shape[1])
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Move data to device
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        
        print("Starting unsupervised training...")
        
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.model(x, edge_index)
            
            # Simple reconstruction loss: try to reconstruct original features
            reconstruction_loss = F.mse_loss(embeddings, x)
            
            # Add diversity regularization to encourage different embeddings
            diversity_loss = -torch.mean(torch.std(embeddings, dim=0))
            total_loss = reconstruction_loss + 0.1 * diversity_loss
            
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:03d}, Loss: {total_loss:.4f}, Recon: {reconstruction_loss:.4f}')
        
        print("Training completed!")
        return self
    
    def get_embeddings(self, data, return_separate=True):
        """Get GraphSAGE embeddings for all nodes"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.model.eval()
        with torch.no_grad():
            x = data.x.to(self.device)
            edge_index = data.edge_index.to(self.device)
            embeddings = self.model(x, edge_index).cpu().numpy()
        
        if return_separate:
            n_diagnostic = data.num_diagnostic
            diagnostic_embeddings = embeddings[:n_diagnostic]
            question_embeddings = embeddings[n_diagnostic:]
            return diagnostic_embeddings, question_embeddings
        else:
            return embeddings

# Example usage function
def create_sample_data():
    """Create sample data in your DataFrame format"""
    data = {
        'diagnostic_action_id': ['D001', 'D001', 'D001', 'D002', 'D002', 'D003', 'D003', 'D003'],
        'diagnostic_action_title': [
            'Blood Pressure Assessment', 'Blood Pressure Assessment', 'Blood Pressure Assessment',
            'Cardiac Evaluation', 'Cardiac Evaluation', 
            'Blood Sugar Testing', 'Blood Sugar Testing', 'Blood Sugar Testing'
        ],
        'question_text': [
            'Do you have chest pain?',
            'Any shortness of breath?', 
            'Do you feel dizzy?',
            'Any palpitations?',
            'Do you have irregular heartbeat?',
            'Do you feel excessively thirsty?',
            'Do you urinate frequently?',
            'Do you feel tired after meals?'
        ],
        'question_id': ['Q001', 'Q002', 'Q003', 'Q004', 'Q005', 'Q006', 'Q007', 'Q008']
    }
    
    return pd.DataFrame(data)

def main_example():
    """Complete example with your DataFrame structure"""
    print("=== Creating Sample Data ===")
    df = create_sample_data()
    print("Sample DataFrame:")
    print(df.head())
    
    print(f"\nData Summary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Unique diagnostics: {df['diagnostic_action_id'].nunique()}")
    print(f"- Unique questions: {df['question_id'].nunique()}")
    
    print("\n=== Processing Data and Creating Graph ===")
    processor = DiagnosticDataProcessor(cosine_threshold=0.3)
    graph_data = processor.process_data(df)
    
    print("\n=== Training GraphSAGE ===")
    graphsage = DiagnosticGraphSAGE(embedding_dim=128, device='cpu')
    graphsage.train_unsupervised(graph_data, num_epochs=50)
    
    print("\n=== Getting GraphSAGE Embeddings ===")
    diagnostic_embeddings, question_embeddings = graphsage.get_embeddings(graph_data)
    
    print(f"Final Results:")
    print(f"- Diagnostic embeddings shape: {diagnostic_embeddings.shape}")
    print(f"- Question embeddings shape: {question_embeddings.shape}")
    
    return processor, graphsage, diagnostic_embeddings, question_embeddings

if __name__ == "__main__":
    # Run the complete example
    processor, graphsage, diag_emb, quest_emb = main_example()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class DrugGeneGNN(nn.Module):
    def __init__(self, num_nodes, embedding_dim=128, hidden_dim=64):
        super(DrugGeneGNN, self).__init__()

        # 1. Düğümler için eğitilebilir başlangıç temsil matrisi (Node Embeddings)
        self.node_emb = nn.Embedding(num_nodes, embedding_dim)

        # 2. GraphSAGE Çizge Evrişim Katmanları
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # 3. İlaç ve Gen düğüm temsilini birleştirip etkileşim skoru üreten sınıflandırıcı
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def encode(self, edge_index):
        """Graf üzerindeki komşuluk mesajlarını toplayarak düğüm temsil vektörlerini günceller."""
        x = self.node_emb.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        """Verilen kaynak (İlaç) ve hedef (Gen) düğüm çiftleri arasındaki bağ olasılığını tahmin eder."""
        src_nodes = z[edge_label_index[0]]
        dst_nodes = z[edge_label_index[1]]

        # İki düğüm vektörünü yan yana birleştiriyoruz (Concat)
        pair_representation = torch.cat([src_nodes, dst_nodes], dim=-1)
        return self.link_predictor(pair_representation).squeeze(-1)

    def forward(self, edge_index, edge_label_index):
        z = self.encode(edge_index)
        return self.decode(z, edge_label_index)



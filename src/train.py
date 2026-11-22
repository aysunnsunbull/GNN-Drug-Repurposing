import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from gnn_model import DrugGeneGNN


def train():
    # 1. Cihaz Ayarı (GPU kullanımı)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim Cihazı: {device}")

    # 2. İşlenmiş Graf Verisini Yükleme
    graph_path = '../data/processed/pytorch_graph.pt'
    try:
        graph_data = torch.load(graph_path, weights_only=False).to(device)
        print(f"Graf Yüklendi: {graph_data.num_nodes} Düğüm, {graph_data.num_edges} Kenar.")
    except FileNotFoundError:
        print("Hata: pytorch_graph.pt dosyası bulunamadı.")
        return

    # 3. Modeli ve Optimizasyon Araçlarını Başlatma
    model = DrugGeneGNN(num_nodes=graph_data.num_nodes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    num_edges = graph_data.edge_index.size(1)
    batch_size = 131072  # 131 binlik paketler GPU bellegini son derece rahat tutar
    epochs = 20

    print("\n--- GNN Eğitimi Başlıyor (Native CUDA Batching) ---")
    model.train()

    for epoch in range(epochs):
        # Her epoch başında kenar indekslerini rastgele karıştırıyoruz
        perm = torch.randperm(num_edges, device=device)
        total_loss = 0.0
        num_batches = 0

        # Veriyi küçük paketler halinde işliyoruz
        for start_idx in range(0, num_edges, batch_size):
            optimizer.zero_grad()

            # GNN ile Düğüm Vektörlerini (Embeddings) Kodlama
            z = model.encode(graph_data.edge_index)

            # Pozitif Kenar Paketi (Gerçek İlaç-Gen Bağları)
            batch_perm = perm[start_idx: start_idx + batch_size]
            pos_edge_batch = graph_data.edge_index[:, batch_perm]

            # Negatif Kenar Paketi (Sahte Bağlar)
            neg_edge_batch = negative_sampling(
                edge_index=graph_data.edge_index,
                num_nodes=graph_data.num_nodes,
                num_neg_samples=pos_edge_batch.size(1)
            ).to(device)

            # Tahminler
            pos_pred = model.decode(z, pos_edge_batch)
            neg_pred = model.decode(z, neg_edge_batch)

            # Etiketler ve Kayıp Hesaplama
            preds = torch.cat([pos_pred, neg_pred])
            labels = torch.cat([
                torch.ones_like(pos_pred),
                torch.zeros_like(neg_pred)
            ])

            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch: {epoch + 1:02d}/{epochs} | Ortalama Kayıp (Loss): {avg_loss:.4f}")

    print("\nEğitim Başarıyla Tamamlandı!")

    # Eğitilmiş modeli ve ağırlıkları kaydetme
    weights_path = '../data/processed/gnn_model_weights.pth'
    torch.save(model.state_dict(), weights_path)
    print(f"Model ağırlıkları başarıyla kaydedildi: {weights_path}")


if __name__ == "__main__":
    train()
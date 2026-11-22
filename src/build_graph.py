import pandas as pd
import torch
from torch_geometric.data import Data


def build_extended_graph():
    print("--- Genişletilmiş Biyolojik Graf Oluşturuluyor ---")

    # 1. Mevcut Verileri Yükleme (CTDBase, DGIdb, GDSC)
    edges_df = pd.read_csv('../data/processed/graph_edges.csv')

    # 2. PathBank Verilerini Yükleme ve İşleme
    print("PathBank verileri okunuyor...")
    try:
        proteins_df = pd.read_csv('../data/pathbank_all_proteins.csv')
    except Exception as e:
        print(f"PathBank dosyası okunurken hata: {e}")
        return

    # PathBank içindeki Gen ve Yolak sütunlarını tespit etme
    gene_col, pathway_col = None, None
    for col in proteins_df.columns:
        if 'gene' in col.lower() or 'symbol' in col.lower():
            gene_col = col
        if 'pathway' in col.lower() and ('name' in col.lower() or 'id' in col.lower() or 'title' in col.lower()):
            pathway_col = col

    if gene_col and pathway_col:
        print(f"PathBank Eşleşmesi: Gen Sütunu -> {gene_col}, Yolak Sütun -> {pathway_col}")

        # Sadece ağımızda halihazırda bulunan genleri filtreleyerek alıyoruz
        existing_genes = set(edges_df['Gene'].unique())
        sub_pathbank = proteins_df[proteins_df[gene_col].isin(existing_genes)].dropna(subset=[gene_col, pathway_col])

        pathbank_edges = pd.DataFrame({
            'Drug': sub_pathbank[gene_col].values,  # Kaynak olarak gen
            'Gene': sub_pathbank[pathway_col].values  # Hedef olarak yolak
        })

        # İlaç-Gen kenarları ile PathBank yolak kenarlarını birleştiriyoruz
        base_edges = edges_df[['Drug', 'Gene']].copy()
        combined_edges = pd.concat([base_edges, pathbank_edges], ignore_index=True)
    else:
        print("Uyarı: PathBank sütunları otomatik bulunamadı, sadece temel ağ ile devam ediliyor.")
        combined_edges = edges_df[['Drug', 'Gene']].copy()

    # 3. Tüm Benzersiz Varlıkları (İlaçlar, Genler, Yolaklar) Tek Bir ID Uzayında Toplama
    unique_nodes = pd.concat([combined_edges['Drug'], combined_edges['Gene']]).unique()
    node_mapping = {node_name: idx for idx, node_name in enumerate(unique_nodes)}

    # DataFrame üzerindeki isimleri sayısal ID'lere dönüştürme
    combined_edges['Src_ID'] = combined_edges['Drug'].map(node_mapping)
    combined_edges['Dst_ID'] = combined_edges['Gene'].map(node_mapping)

    # Güncellenmiş kenar listesini kaydediyoruz
    combined_edges.to_csv('../data/processed/graph_edges.csv', index=False)
    print(f"Toplam Düğüm Sayısı (İlaç + Gen + Yolak): {len(node_mapping)}")
    print(f"Toplam Kenar Sayısı: {len(combined_edges)}")

    # 4. PyTorch Geometric Tensor Formatına Dönüştürme
    src_ids = torch.tensor(combined_edges['Src_ID'].values, dtype=torch.long)
    dst_ids = torch.tensor(combined_edges['Dst_ID'].values, dtype=torch.long)

    # Yönlü olmayan (undirected) graf yapısı için çift yönlü kenar ekleme
    edge_index = torch.stack([
        torch.cat([src_ids, dst_ids], dim=0),
        torch.cat([dst_ids, src_ids], dim=0)
    ], dim=0)

    graph_data = Data(edge_index=edge_index, num_nodes=len(node_mapping))

    # PyTorch graf dosyasını kaydediyoruz
    torch.save(graph_data, '../data/processed/pytorch_graph.pt')
    print("Genişletilmiş PyTorch Grafı başarıyla kaydedildi: pytorch_graph.pt")


if __name__ == "__main__":
    build_extended_graph()
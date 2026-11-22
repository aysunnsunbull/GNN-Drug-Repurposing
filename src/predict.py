import torch
import pandas as pd
from gnn_model import DrugGeneGNN


def find_top_drugs_for_gene(gene_name, top_k=10):
    print(f"\n--- '{gene_name.upper()}' Geni İçin YENİ İlaç Keşfi Başlatılıyor ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Ağ Verilerini Yükleme
    edges_df = pd.read_csv('../data/processed/graph_edges.csv')
    graph_data = torch.load('../data/processed/pytorch_graph.pt', weights_only=False).to(device)

    # 2. Filtreleme İçin GDSC (Gerçek İlaçlar) Listesini Yükleme
    print("Filtre 1: Terapötik İlaç Listesi Yükleniyor...")
    gdsc_df = pd.read_excel('../data/GDSC2_fitted_dose_response_27Oct23.xlsx')
    # Sadece GDSC'deki ilaç isimlerini büyük harfe çevirerek bir kümeye (set) alıyoruz
    valid_therapeutic_drugs = set(gdsc_df['DRUG_NAME'].astype(str).str.upper().unique())

    # 3. İsim-ID Eşleştirmeleri
    drug_mapping = dict(zip(edges_df['Drug'], edges_df['Src_ID']))
    gene_mapping = dict(zip(edges_df['Gene'], edges_df['Dst_ID']))
    reverse_drug_mapping = {v: k for k, v in drug_mapping.items()}

    gene_name = gene_name.upper()
    if gene_name not in gene_mapping:
        print(f"Hata: '{gene_name}' eğitim ağımızda bulunamadı.")
        return

    target_gene_id = gene_mapping[gene_name]

    # Filtre 2: Bu gen için CTDBase/DGIdb'de ZATEN BİLİNEN ilaçları listeye alıyoruz
    known_interactions = set(edges_df[edges_df['Gene'] == gene_name]['Drug'].values)

    # 4. Eğitilmiş Modeli Yükleme
    model = DrugGeneGNN(num_nodes=graph_data.num_nodes).to(device)
    model.load_state_dict(torch.load('../data/processed/gnn_model_weights.pth', weights_only=True))
    model.eval()

    print("Filtre 2: Bilinen Etkileşimler Eleniyor ve Tahmin Yapılıyor...")
    # 5. Tüm İlaçları Eşzamanlı (Batch) Test Etme
    with torch.no_grad():
        z = model.encode(graph_data.edge_index)

        all_drug_ids = torch.tensor(list(drug_mapping.values()), dtype=torch.long).to(device)
        target_gene_ids = torch.full_like(all_drug_ids, target_gene_id)

        edge_label_index = torch.stack([all_drug_ids, target_gene_ids], dim=0)

        # İşlem bitince filtreleme yapabilmek için sonuçları CPU'ya alıyoruz
        probabilities = model.decode(z, edge_label_index).cpu().numpy()
        all_drug_ids_cpu = all_drug_ids.cpu().numpy()

    # 6. Sonuçları Filtreleme ve Sıralama
    discovery_results = []

    for i in range(len(all_drug_ids_cpu)):
        drug_id = all_drug_ids_cpu[i]
        drug_name_str = reverse_drug_mapping[drug_id]
        prob = probabilities[i]

        # ŞART 1: Sadece tedavi edici (GDSC'de olan) bir ilaç mı?
        if drug_name_str not in valid_therapeutic_drugs:
            continue

        # ŞART 2: Daha önce BİLİNMEYEN yeni bir keşif mi?
        if drug_name_str in known_interactions:
            continue

        discovery_results.append((drug_name_str, prob))

    # Olasılığa (İhtimale) göre büyükten küçüğe sıralıyoruz
    discovery_results.sort(key=lambda x: x[1], reverse=True)

    # 7. Ekrana Yazdırma
    print(f"\n[{gene_name}] Geni İçin Önerilen YENİ {top_k} Potansiyel İlaç Adayı:")
    print("-" * 70)
    for i in range(min(top_k, len(discovery_results))):
        drug_name_str, prob = discovery_results[i]
        print(f"{i + 1:02d}. Aday İlaç: {drug_name_str:<25} | Etkileşim İhtimali: %{prob * 100:.2f}")


if __name__ == "__main__":
    # Test: Meme Kanseri (BRCA1) ve Alzheimer (APOE) genleri için tarama yapalım
    find_top_drugs_for_gene("BRCA1", top_k=10)
    find_top_drugs_for_gene("APOE", top_k=10)
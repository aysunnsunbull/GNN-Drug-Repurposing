import pandas as pd
import os

def load_and_clean_data():
    print("Veri setleri yükleniyor ve ön temizlik yapılıyor...\n")

    # 1. CTDBase Verileri Yükleniyor
    print("--- CTDBase ---")
    ctd_chem_gene = pd.read_csv('../data/CTD_chem_gene_ixns.csv', skiprows=27)
    ctd_chem_disease = pd.read_csv('../data/CTD_curated_chemicals_diseases.csv', skiprows=27)
    ctd_gene_disease = pd.read_csv('../data/CTD_curated_genes_diseases.csv', skiprows=27)

    # CTDBase sütun başlıklarındaki '# ' karakterlerini siliyoruz
    for df in [ctd_chem_gene, ctd_chem_disease, ctd_gene_disease]:
        df.rename(columns=lambda x: x.replace('# ', ''), inplace=True)

    print(f"Temizlenmiş İlaç-Gen Sütunları: {list(ctd_chem_gene.columns)[:5]}")

    # 2. GDSC Verisi Yükleniyor
    print("\n--- GDSC ---")
    gdsc_df = pd.read_excel('../data/GDSC2_fitted_dose_response_27Oct23.xlsx')
    print(f"Satır Sayısı: {gdsc_df.shape[0]}")

    # 3. DGIdb Verisi Yükleniyor (TSV formatı)
    print("\n--- DGIdb ---")
    dgidb_df = pd.read_csv('../data/interactions.tsv', sep='\t', on_bad_lines='skip', low_memory=False, skiprows=2)
    print(f"Satır Sayısı: {dgidb_df.shape[0]}")
    print(f"Sütun İsimleri: {list(dgidb_df.columns)[:5]}\n")

    return ctd_chem_gene, ctd_chem_disease, ctd_gene_disease, gdsc_df, dgidb_df


if __name__ == "__main__":
    ctd_cg, ctd_cd, ctd_gd, gdsc, dgidb = load_and_clean_data()
    print("Tüm veriler başarıyla yüklendi ve temel temizlik tamamlandı!")


def build_knowledge_graph_edges(ctd_cg, dgidb_df):
    print("\n--- GNN Ağ Kenarları (Edges) Oluşturuluyor ---")

    # 1. CTDBase İlaç-Gen verisini hazırlama
    # CTDBase'deki gen sembollerini anahtar olarak alıyoruz
    ctd_edges = ctd_cg[['ChemicalName', 'GeneSymbol']].copy()
    ctd_edges.columns = ['Drug', 'Gene']
    print(f"CTDBase'den alınan İlaç-Gen bağı sayısı: {ctd_edges.shape[0]}")

    # 2. DGIdb İlaç-Gen verisini hazırlama
    # DGIdb'deki ilaç ve gen isimlerini alıp aynı formata getiriyoruz
    # NaN (boş) verileri düşürüyoruz
    dgidb_edges = dgidb_df[['drug_claim_name', 'gene_name']].dropna().copy()
    dgidb_edges.columns = ['Drug', 'Gene']

    # DGIdb'deki ilaç ve gen isimlerini büyük harfe çevirerek standartlaştırıyoruz
    dgidb_edges['Drug'] = dgidb_edges['Drug'].str.upper()
    dgidb_edges['Gene'] = dgidb_edges['Gene'].str.upper()
    ctd_edges['Drug'] = ctd_edges['Drug'].str.upper()
    ctd_edges['Gene'] = ctd_edges['Gene'].str.upper()

    print(f"DGIdb'den alınan temiz İlaç-Gen bağı sayısı: {dgidb_edges.shape[0]}")

    # 3. Ağları Birleştirme (Union)
    # İki veri tabanından gelen ilişkileri tek bir tabloda alt alta ekliyoruz
    combined_edges = pd.concat([ctd_edges, dgidb_edges])

    # Aynı ilaç-gen etkileşimini iki kez modele vermemek için tekrarları siliyoruz (Drop Duplicates)
    final_edges = combined_edges.drop_duplicates()

    print(f"Birleştirilmiş ve Tekrarsız (Unique) Toplam İlaç-Gen Bağı (Edge) Sayısı: {final_edges.shape[0]}")
    return final_edges


# if __name__ bloğunu güncelliyoruz:
if __name__ == "__main__":
    ctd_cg, ctd_cd, ctd_gd, gdsc, dgidb = load_and_clean_data()
    print("Tüm veriler başarıyla yüklendi ve temel temizlik tamamlandı!\n")

    # Yeni birleştirme fonksiyonumuzu çağırıyoruz
    drug_gene_edges = build_knowledge_graph_edges(ctd_cg, dgidb)


def check_gdsc_overlap(gdsc_df, final_edges):
    print("\n--- GDSC Verisi ile Ağın Kesişimi Kontrol Ediliyor ---")

    # GDSC standart formatında ilaç isimleri genelde 'DRUG_NAME' sütunundadır.
    # Önce tam sütun isimlerini görelim:
    print(f"GDSC Sütunları: {list(gdsc_df.columns)[:8]}...")

    if 'DRUG_NAME' in gdsc_df.columns:
        # GDSC'deki ilaç isimlerini de büyük harfe çevirip standartlaştırıyoruz
        gdsc_df['DRUG_NAME'] = gdsc_df['DRUG_NAME'].astype(str).str.upper()

        # Ağımızdaki (CTDBase+DGIdb) benzersiz ilaçları alıyoruz
        graph_drugs = set(final_edges['Drug'].unique())

        # GDSC'deki benzersiz ilaçları alıyoruz
        gdsc_drugs = set(gdsc_df['DRUG_NAME'].unique())

        # İki kümenin kesişimini (ortak ilaçları) buluyoruz
        common_drugs = graph_drugs.intersection(gdsc_drugs)

        print(f"Oluşturduğumuz Ağdaki Benzersiz İlaç Sayısı: {len(graph_drugs)}")
        print(f"GDSC Veri Setindeki Benzersiz İlaç Sayısı: {len(gdsc_drugs)}")
        print(f"Eşleşen (Hem Ağda Hem GDSC'de Olan) İlaç Sayısı: {len(common_drugs)}")
    else:
        print("GDSC tablosunda 'DRUG_NAME' sütunu bulunamadı. Lütfen sütun isimlerini kontrol edin.")


def create_node_mappings_and_save(final_edges, gdsc_df):
    print("\n--- Düğüm İndeksleri (Node IDs) Oluşturuluyor ve Kaydediliyor ---")

    # 0. Boş (NaN) değerleri temizleyip metin tipine dönüştürüyoruz (TypeError önlemi)
    final_edges = final_edges.dropna(subset=['Drug', 'Gene']).copy()
    final_edges['Drug'] = final_edges['Drug'].astype(str)
    final_edges['Gene'] = final_edges['Gene'].astype(str)

    # 1. Tüm benzersiz Düğümleri (İlaçlar ve Genler) Topluyoruz
    unique_drugs = sorted(list(final_edges['Drug'].unique()))
    unique_genes = sorted(list(final_edges['Gene'].unique()))

    # Düğümlere benzersiz integer ID veriyoruz
    drug_to_id = {drug: i for i, drug in enumerate(unique_drugs)}
    # Gen ID'leri ilaç ID'lerinin bittiği yerden devam ettiriyoruz
    gene_to_id = {gene: i + len(unique_drugs) for i, gene in enumerate(unique_genes)}

    # 2. Kenarları (Edges) İsimlerden ID'lere Dönüştürüyoruz
    final_edges['Src_ID'] = final_edges['Drug'].map(drug_to_id)
    final_edges['Dst_ID'] = final_edges['Gene'].map(gene_to_id)

    print(
        f"Toplam Düğüm (Node) Sayısı: {len(drug_to_id) + len(gene_to_id)} (İlaç: {len(drug_to_id)}, Gen: {len(gene_to_id)})")
    print(f"Toplam Kenar (Edge) Sayısı: {len(final_edges)}")

    # 3. İşlenmiş Verileri Kaydetme
    processed_dir = '../data/processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Kenar listesini CSV olarak kaydediyoruz
    edges_path = os.path.join(processed_dir, 'graph_edges.csv')
    final_edges[['Src_ID', 'Dst_ID', 'Drug', 'Gene']].to_csv(edges_path, index=False)

    print(f"\nTemizlenmiş Graf Verisi Başarıyla Kaydedildi: {edges_path}")

if __name__ == "__main__":
    # 1. Verileri Yükle
    ctd_cg, ctd_cd, ctd_gd, gdsc, dgidb = load_and_clean_data()

    # 2. GNN Ağını Kur
    drug_gene_edges = build_knowledge_graph_edges(ctd_cg, dgidb)

    # 3. GDSC Kesişimini Kontrol Et
    check_gdsc_overlap(gdsc, drug_gene_edges)

    # 4. ID Haritasını Çıkar ve İşlenmiş Ağ Verisini Kaydet
    create_node_mappings_and_save(drug_gene_edges, gdsc)
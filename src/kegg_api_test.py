import requests


def test_kegg_api():
    print("KEGG API'sine bağlanılıyor...")
    # KEGG üzerinden İnsan (Homo sapiens - hsa) biyolojik yolaklarının (pathways) listesini çekiyoruz
    url = "http://rest.kegg.jp/list/pathway/hsa"

    try:
        response = requests.get(url)

        if response.status_code == 200:
            print("Bağlantı Başarılı! İşte ilk 5 veri:\n")
            # Gelen veriyi satırlara bölüp ilk 5 tanesini ekrana yazdırıyoruz
            lines = response.text.strip().split('\n')
            for line in lines[:5]:
                print(line)
        else:
            print(f"Hata oluştu! Durum Kodu: {response.status_code}")

    except Exception as e:
        print(f"Bir hata meydana geldi: {e}")


if __name__ == "__main__":
    test_kegg_api()
import json
import random
from tqdm import tqdm
from typing import List, Dict

# Konular ve ilgili kelime listeleri
TOPICS = {
    "okul": {
        "yerler": ["okula", "kütüphaneye", "laboratuvara", "dersliğe"],
        "fiiller": ["gidiyor musun", "geldin mi", "çalışıyor musun", "katıldın mı"],
        "cevaplar": ["Evet, {yer} gidiyorum.", "Hayır, bugün gitmiyorum.", "Henüz gitmedim, ama gideceğim.", "{yer} geldim."],
        "kisiler": ["Ben", "Sen", "O", "Biz", "Siz", "Onlar"]
    },
    "ev": {
        "yerler": ["eve", "mutfağa", "salona", "bahçeye"],
        "fiiller": ["geldin mi", "yemek yedin mi", "dinleniyor musun", "uyuyor musun"],
        "cevaplar": ["Evet, {yer} geldim.", "Hayır, henüz gitmedim.", "Evet, yedim.", "Henüz yemek yemedim."],
        "kisiler": ["Ben", "Sen", "O", "Biz", "Siz", "Onlar"]
    },
    "iş": {
        "yerler": ["işe", "toplantıya", "ofise", "şirkete"],
        "fiiller": ["geldin mi", "başladın mı", "çalışıyor musun", "yetiştirdin mi"],
        "cevaplar": ["Evet, {yer} geldim.", "Henüz başlamadım.", "Evet, çalışıyorum.", "Evet, işi bitirdim."],
        "kisiler": ["Ben", "Sen", "O", "Biz", "Siz", "Onlar"]
    },
    "market": {
        "yerler": ["markete", "bakkala", "manava", "kasaba"],
        "fiiller": ["gittin mi", "aldın mı", "bakıyorsun mu", "çalışıyor mu"],
        "cevaplar": ["Evet, {yer} gittim.", "Hayır, henüz gitmedim.", "Evet, aldım.", "Hayır, almıyorum."],
        "kisiler": ["Ben", "Sen", "O", "Biz", "Siz", "Onlar"]
    },
    "seyahat": {
        "yerler": ["istasyona", "havaalanına", "otobüse", "tren garına"],
        "fiiller": ["gittin mi", "bilet aldın mı", "hazırlandın mı", "geldin mi"],
        "cevaplar": ["Evet, {yer} gittim.", "Hayır, henüz gitmedim.", "Evet, bilet aldım.", "Henüz hazır değilim."],
        "kisiler": ["Ben", "Sen", "O", "Biz", "Siz", "Onlar"]
    }
}

# Şablonlar soru-cümlesi şeklinde ve cevap için kalıplar
# prompt = "{Kişi} {yer} {fiil}?"
# response = cevaplardan biri (değişkenlere uygun şekilde)

def generate_dialogue(topic_data) -> Dict[str, str]:
    kisi = random.choice(topic_data["kisiler"])
    yer = random.choice(topic_data["yerler"])
    fiil = random.choice(topic_data["fiiller"])
    
    # Soru (prompt)
    prompt = f"{kisi} {yer} {fiil}?"
    
    # Cevap (response)
    cevap_template = random.choice(topic_data["cevaplar"])
    response = cevap_template.format(yer=yer)
    
    return {"prompt": prompt, "response": response}

def generate_dataset(total: int) -> List[Dict[str, str]]:
    dialogues = []
    topic_keys = list(TOPICS.keys())
    
    for _ in tqdm(range(total), desc="Diyalog üretiliyor"):
        topic = random.choice(topic_keys)
        dialogue = generate_dialogue(TOPICS[topic])
        dialogues.append(dialogue)
    
    return dialogues

def main():
    TOTAL_DATA = 100_000  # 100 bin
    
    data = generate_dataset(TOTAL_DATA)
    
    # JSON dosyasına yaz
    with open("synthetic_turkish_dialogues.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"{TOTAL_DATA} adet sentetik Türkçe diyalog üretildi ve 'synthetic_turkish_dialogues.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    main()

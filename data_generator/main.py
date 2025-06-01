import json
import random
import uuid
import time
import os
import concurrent.futures
from typing import Optional, List, Dict, Set, Tuple
from ollama import Client
from tqdm import tqdm
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import glob
from topics import TOPICS  # Senin mevcut topics modülün

OLLAMA_MODEL = "gemma3:1b"
OUTPUT_FILE = "generated_dialogues.json"
TEMP_OUTPUT_PREFIX = "temp_generated_dialogues_"
TOTAL_TARGET_DIALOGUES = 10000
MAX_RETRIES = 5
MAX_WORKERS = 8
BATCH_SIZE = 200
DELAY_AFTER_FAILURE = 2.0
SAVE_INTERVAL_SECONDS = 60

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dialogue_generator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

client = Client(host='http://localhost:11434')

class DialogueGenerator:
    def __init__(self):
        self.existing_dialogues: List[Dict] = []
        self.unique_dialogue_pairs: Set[Tuple[str, str]] = set()
        self.current_dialogue_count = 0
        self.topic_stats = defaultdict(int)
        self.last_save_time = time.time()
        self.buffer: List[Dict] = []
        
    def load_existing_data(self, filepath: str) -> None:
        try:
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.existing_dialogues = json.load(f)
                    for item in self.existing_dialogues:
                        if 'prompt' in item and 'response' in item:
                            self.unique_dialogue_pairs.add((item['prompt'], item['response']))
                    self.current_dialogue_count = len(self.existing_dialogues)
                    logging.info(f"Mevcut {self.current_dialogue_count} diyalog çifti yüklendi.")
                    
                    for item in self.existing_dialogues:
                        if 'topic' in item:
                            self.topic_stats[item['topic']] += 1
        except Exception as e:
            logging.error(f"Veri yükleme hatası: {e}")
            raise

    def save_data(self, filepath: str, temp_filepath_prefix: str, force: bool = False) -> None:
        current_time = time.time()
        if not force and current_time - self.last_save_time < SAVE_INTERVAL_SECONDS and self.buffer:
            return
            
        if not self.buffer:
            return
            
        try:
            temp_filepath = f"{temp_filepath_prefix}{uuid.uuid4()}.json"
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.existing_dialogues + self.buffer, f, ensure_ascii=False, indent=2)
            
            os.replace(temp_filepath, filepath)
            
            self.existing_dialogues.extend(self.buffer)
            for item in self.buffer:
                if 'topic' in item:
                    self.topic_stats[item['topic']] += 1
            self.buffer.clear()
            
            self.last_save_time = current_time
            logging.info(f"Veri kaydedildi. Toplam: {len(self.existing_dialogues)}")
        except Exception as e:
            logging.error(f"Veri kaydetme hatası: {e}")
            raise

    def generate_ollama_response(self, prompt: str) -> Optional[str]:
        for attempt in range(MAX_RETRIES):
            try:
                messages = [
                    {'role': 'system', 'content': 'Sen faydalı bir sohbet botusun. Sana verilen promptlara uygun, doğal ve akıcı Türkçe ile cevap ver.'},
                    {'role': 'user', 'content': prompt}
                ]
                
                response = client.chat(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    options={
                        'temperature': random.uniform(0.6, 0.8),
                        'top_k': random.randint(30, 50),
                        'top_p': random.uniform(0.8, 0.95),
                        'num_ctx': 2048,
                        'repeat_penalty': random.uniform(1.1, 1.3)
                    },
                    stream=False
                )
                
                return response['message']['content'].strip()
            except Exception as e:
                logging.warning(f"Ollama API hatası (Deneme {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(DELAY_AFTER_FAILURE * (attempt + 1))
        return None

    def generate_dialogue(self, topic: str, prompt: str) -> Optional[Dict]:
        try:
            response = self.generate_ollama_response(prompt)
            if response:
                dialogue_pair = (prompt, response)
                if dialogue_pair not in self.unique_dialogue_pairs:
                    return {
                        "prompt": prompt,
                        "response": response,
                        "topic": topic,
                        "uuid": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat()
                    }
            return None
        except Exception as e:
            logging.error(f"Diyalog oluşturma hatası: {e}")
            return None

    def select_topic_prompt(self) -> Tuple[str, str]:
        least_used = sorted(self.topic_stats.items(), key=lambda x: x[1])[:3]
        least_used_topics = [topic for topic, _ in least_used] or list(TOPICS.keys())
        
        selected_topic = random.choices(
            population=list(TOPICS.keys()),
            weights=[0.7 if t in least_used_topics else 0.3 for t in TOPICS.keys()],
            k=1
        )[0]
        
        return selected_topic, random.choice(TOPICS[selected_topic])

    def cleanup_temp_files(self, prefix: str):
        for f in glob.glob(f"{prefix}*.json"):
            try:
                os.remove(f)
                logging.info(f"Geçici dosya silindi: {f}")
            except Exception as e:
                logging.error(f"Geçici dosya silinirken hata oluştu {f}: {e}")

    def run(self, total_target: int) -> None:
        start_time = time.time()
        
        with tqdm(total=total_target, initial=self.current_dialogue_count) as pbar:
            while self.current_dialogue_count < total_target:
                # İşçi havuzu
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = []
                    while len(futures) < MAX_WORKERS * 2 and self.current_dialogue_count + len(futures) < total_target:
                        topic, prompt = self.select_topic_prompt()
                        futures.append(executor.submit(self.generate_dialogue, topic, prompt))
                    
                    if not futures:
                        time.sleep(0.5)
                        continue

                    done, _ = concurrent.futures.wait(futures, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for future in done:
                        result = future.result()
                        if result:
                            self.buffer.append(result)
                            self.unique_dialogue_pairs.add((result['prompt'], result['response']))
                            self.current_dialogue_count += 1
                            pbar.update(1)
                            
                            # Yüzdeyi manuel güncelle
                            percentage = (pbar.n / total_target) * 100
                            pbar.set_description(f"Diyaloglar Üretiliyor %{percentage:.5f}")

                            if len(self.buffer) >= BATCH_SIZE:
                                self.save_data(OUTPUT_FILE, TEMP_OUTPUT_PREFIX)
                        
                        futures.remove(future)
                    
                    self.save_data(OUTPUT_FILE, TEMP_OUTPUT_PREFIX)
                    
                    time.sleep(0.1)
            
        if self.buffer:
            self.save_data(OUTPUT_FILE, TEMP_OUTPUT_PREFIX, force=True)
            
        elapsed = timedelta(seconds=time.time() - start_time)
        logging.info(f"\n--- İşlem Tamamlandı ---")
        logging.info(f"Toplam Süre: {elapsed}")
        logging.info(f"Toplam Diyalog: {self.current_dialogue_count}")
        logging.info("Konu Dağılımı:")
        for topic, count in sorted(self.topic_stats.items(), key=lambda x: x[1], reverse=True):
            if self.current_dialogue_count > 0:
                logging.info(f"  {topic}: {count} ({count/self.current_dialogue_count:.1%})")
            else:
                logging.info(f"  {topic}: {count}")

def main():
    logging.info(f"Ollama ile veri üretimi başlatılıyor. Model: {OLLAMA_MODEL}")
    
    generator = DialogueGenerator()
    
    try:
        generator.load_existing_data(OUTPUT_FILE)
        
        if generator.current_dialogue_count > 0:
            overwrite_choice = input(f"'{OUTPUT_FILE}' dosyası mevcut. Üzerine yazmak mı (y) yoksa mevcut verilere eklemek mi (e)? [e/y]: ").lower()
            if overwrite_choice == 'y':
                generator.existing_dialogues = []
                generator.unique_dialogue_pairs = set()
                generator.current_dialogue_count = 0
                generator.topic_stats = defaultdict(int)
                logging.info(f"'{OUTPUT_FILE}' dosyası sıfırlandı.")
                generator.cleanup_temp_files(TEMP_OUTPUT_PREFIX)
            else:
                logging.info("Mevcut verilere ekleme yapılacak.")
        
        generator.run(TOTAL_TARGET_DIALOGUES)
        
    except KeyboardInterrupt:
        logging.info("Kullanıcı tarafından durduruldu. Mevcut veriler kaydediliyor...")
        if generator.buffer:
            generator.save_data(OUTPUT_FILE, TEMP_OUTPUT_PREFIX, force=True)
    except Exception as e:
        logging.error(f"Kritik hata: {e}")
        if generator.buffer:
            generator.save_data(OUTPUT_FILE, TEMP_OUTPUT_PREFIX, force=True)
    finally:
        logging.info("Program sonlandırıldı.")
        generator.cleanup_temp_files(TEMP_OUTPUT_PREFIX)


if __name__ == "__main__":
    main()

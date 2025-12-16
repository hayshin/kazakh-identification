import os
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.models.base import Model, LangDetectorChoices, Language

model_path = os.path.join('data', 'lid_model')

print(f"Загружаю модель из {model_path}...")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Переводим в режим оценки (обязательно!)
model.eval()

# Если есть GPU, кидаем туда (для скорости), если нет - на CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 2. Функция для предсказания с процентами
def predict_language(text, temperature=2):
    """
    temperature: 
        1.0 - стандартная уверенность (как раньше)
        >1.0 - смягченная уверенность (покажет 70/30 вместо 99/0)
    """
    # Токенизация
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # Отправляем на девайс
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Получаем логиты
    logits = outputs.logits
    
    # --- ТРЮК С ТЕМПЕРАТУРОЙ ---
    # Делим логиты на число > 1. Это "сплющивает" распределение вероятностей.
    # Разница между KK и RU становится меньше.
    scaled_logits = logits / temperature
    
    # Softmax уже от масштабированных логитов
    probs = F.softmax(scaled_logits, dim=-1) # [prob_KK, prob_RU]
    
    # Достаем значения
    prob_kk = probs[0][0].item() * 100
    prob_ru = probs[0][1].item() * 100
    
    # Определяем победителя (он не изменится от температуры, только % изменится)
    predicted_label = "kazakh" if prob_kk > prob_ru else "russian"
    
    return {
        "primary_lang": predicted_label, 
        "kazakh": prob_kk, 
        "russian": prob_ru,
        "other": 0.00
    }




class KazNLP(Model):
    def detect_lang_single(self, txt: str) -> Language:
        predict_language(txt)["primary_lang"]

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        predict_language(txt)

        

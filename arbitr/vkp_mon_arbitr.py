# Автоматический пересказ решений арбитражных судов
# Требуется аккаунт "Контур.Фокуса" для работы

from vkp_pdf import getpdf  # Обработка PDF-файлов
import vkp_db as db  # Логика работы с базой данных
import user_data  # Заголовки запроса, включая cookies

from gpt4all import GPT4All  # Поддержка ИИ

import requests
from bs4 import BeautifulSoup
from time import sleep
from tqdm import tqdm
import logging
from urllib.parse import urljoin

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arbitr_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def safe_request(url, max_retries=3, retry_delay=5):
    """Безопасный запрос с повторными попытками"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=user_data.headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {str(e)}")
            if attempt < max_retries - 1:
                sleep(retry_delay * (attempt + 1))
                continue
            logger.error(f"Не удалось выполнить запрос к {url} после {max_retries} попыток")
            raise

def process_document(link, model):
    """Обработка одного документа"""
    try:
        # Проверяем, был ли уже загружен документ
        if db.indb(link):
            logger.info(f"Документ {link} уже в архиве, пропускаем")
            return None
        
        # Загружаем текст из PDF
        text = getpdf(link, {}, user_data.headers)
        if not text:
            logger.error(f"Не удалось извлечь текст из {link}")
            return None
        
        # Проверка релевантности документа
        if not any(substring in text for substring in ['Санкт-Петербург', 'СПб', 'Ленинград']):
            logger.info(f"Документ {link} не касается Петербурга")
            return None
        
        if not any(substring in text for substring in ['ешение', 'остановление', 'пределение']):
            logger.info(f"Документ {link} не является судебным актом")
            return None
        
        # Подготовка текста для GPT
        text_length = len(text)
        if text_length > 2000:
            text4gpt = f"Перескажи коротко текст c указанием истца, ответчика, сути дела и его итогов: {text[:1000]} {text[-1000:]}"
        else:
            text4gpt = f"Перескажи коротко текст c указанием истца, ответчика, сути дела и его итогов: {text[:2000]}"
        
        logger.info(f"Обработка документа длиной {text_length} символов")
        
        # Запрос к GPT
        logger.info("Отправка запроса к GPT...")
        txt = ''
        with model.chat_session():
            g = model.generate(text4gpt, max_tokens=256, streaming=True)
            for i, v in enumerate(g):
                txt += v
        
        logger.info(f"Получен ответ длиной {len(txt)} символов")
        
        # Сохранение в БД
        db.todb('arbitr', link, txt)
        logger.info(f"Документ {link} успешно обработан и сохранен")
        
        return txt
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа {link}: {str(e)}", exc_info=True)
        return None

def main():
    """Основная функция"""
    try:
        logger.info('Загрузка модели GPT')
        model = GPT4All(user_data.model_path, allow_download=False)
        logger.info('Модель загружена')
        
        # Получаем список документов
        logger.info('Загрузка списка документов с focus.kontur.ru')
        response = safe_request('https://focus.kontur.ru/content/mon')
        soup = BeautifulSoup(response.content, "lxml")
        
        # Находим все ссылки на документы
        documents = soup.find_all('a', {'class': 'hover-underline org-changes-document'})
        logger.info(f"Найдено {len(documents)} документов для обработки")
        
        # Обрабатываем документы
        for doc in tqdm(documents, desc="Обработка документов"):
            try:
                link = urljoin('https://focus.kontur.ru', doc['href'])
                logger.info(f"Обработка документа: {link}")
                process_document(link, model)
                sleep(1)  # Пауза между запросами
            except Exception as e:
                logger.error(f"Ошибка при обработке документа: {str(e)}", exc_info=True)
                continue
                
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
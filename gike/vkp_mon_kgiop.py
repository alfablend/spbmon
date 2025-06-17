# Модуль проверки появления новых историко-культурных экспертиз
# на сайте КГИОП и их краткого пересказа

import sqlite3
import pandas as pd
import ocrmypdf
import user_data # Заголовки запроса, включая cookies
import json

from gpt4all import GPT4All #Поддержка ИИ

import requests
from bs4 import BeautifulSoup  
from time import sleep
from random import randint
from tqdm import tqdm
import datetime
import re
import traceback
import logging
import logging.handlers

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import uuid

import os
from PIL import Image
import io

from pypdf import PdfReader
from pdf2image import convert_from_path

# Настройка логирования
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Ротация логов по размеру (1 МБ)
    log_handler = logging.handlers.RotatingFileHandler(
        'kgiop_monitor.log',
        maxBytes=1024*1024,  # 1 MB
        backupCount=3,
        encoding='utf-8'
    )
    log_handler.setFormatter(log_formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(log_handler)
    
    # Также выводим логи в консоль
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    
    return logger


###
### Работа с базой данных    
###
def indb(link): 
    try:
        with sqlite3.connect("kgiop_db.db") as con:
            cur = con.execute("SELECT link FROM kgiop WHERE link = ?", (link,))
        return link in str(cur.fetchall())
    except Exception as e:
        logger.error(f"Ошибка в функции indb: {str(e)}")
        return False
 
def todb(link, summary, text, map_filename, thumb_filename): 
    try:
        #Храним только 1 тыс. последних ссылок
        date = datetime.datetime.now()
        with sqlite3.connect("kgiop_db.db") as con:
            con.execute("DELETE FROM kgiop WHERE date not in (SELECT date FROM kgiop ORDER BY DATE DESC LIMIT 1000);")         
            con.execute("INSERT INTO kgiop VALUES (?, ?, ?, ?, ?, ?);", 
                       (date, link, summary, text, map_filename, thumb_filename))
            df = pd.read_sql_query("SELECT date, link, summary, map_filename, thumb_filename FROM kgiop;", con)
            
            # Модифицируем HTML для отображения картинок
            def make_image_html(map_filename):
                if map_filename:
                    return f'<img src="maps/{map_filename}" width="300">'
                return ''
            
            def make_thumb_html(thumb_filename):
                if thumb_filename:
                    return f'<img src="thumbnails/{thumb_filename}" width="300">'
                return ''
            
            df['Карта'] = df['map_filename'].apply(make_image_html)
            df['Миниатюры'] = df['thumb_filename'].apply(make_thumb_html)
            df = df.drop(columns=['map_filename', 'thumb_filename'])
            
            full_html = df.to_html(escape=False)
            with open("db_html.html", "w", encoding="utf-8") as f:
                f.write(full_html)
        
        logger.info(f"Данные сохранены в БД для ссылки: {link}")
    except Exception as e:
        logger.error(f"Ошибка в функции todb: {str(e)}")
        raise

def extract_text_fragments(text, keywords, max_fragment_length=200, total_max_length=100000):
    """
    Извлекает фрагменты текста (начиная с конца текста), начинающиеся с заданных ключевых слов,
    и заканчивающиеся точкой, затем объединяет их в один текст с ограничением длины.
    
    :param text: исходный текст для поиска
    :param keywords: список ключевых слов
    :param max_fragment_length: максимальная длина каждого фрагмента
    :param total_max_length: максимальная длина итогового текста
    :return: объединенный текст из найденных фрагментов (в обратном порядке)
    """
    try:
        # Создаем regex паттерн для поиска ключевых слов с разными окончаниями и регистром
        pattern = r'(?:' + '|'.join(
            r'(?:\b' + re.escape(keyword.lower()) + r'\w*\b)'
            for keyword in keywords
        ) + r')'
        
        # Ищем все вхождения ключевых слов (без учета регистра)
        fragments = []
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start_pos = match.start()
            
            # Ищем следующую точку после ключевого слова
            end_pos = text.find('.', start_pos)
            if end_pos == -1:
                end_pos = len(text)
            else:
                end_pos += 1  # включая точку
            
            # Ограничиваем длину фрагмента
            fragment = text[start_pos:end_pos].strip()
            if len(fragment) > max_fragment_length:
                fragment = fragment[:max_fragment_length] + '...'
            
            fragments.append((start_pos, fragment))
        
        # Сортируем фрагменты в обратном порядке (по позиции в тексте)
        fragments.sort(reverse=True, key=lambda x: x[0])
        
        # Объединяем фрагменты, соблюдая ограничение общей длины
        result = []
        total_length = 0
        for _, fragment in fragments:
            if total_length + len(fragment) + 1 <= total_max_length:  # +1 для пробела
                result.append(fragment)
                total_length += len(fragment) + 1
            else:
                break
        
        return ' '.join(result)
    except Exception as e:
        logger.error(f"Ошибка в функции extract_text_fragments: {str(e)}")
        return ""

def get_gpt_text(text):
    try:
        if len(text) > 2000:
            text4gpt = f"О каком объекте идет речь, по какому адресу, кто собственник, кто заказчик, какие работы предусматриваются, какой вывод экспертизы: {text[:1000]} {text[-1000:]}"
            logger.info(f"Длина текста для GPT: {len(text4gpt)}")
        else:
            text4gpt = f"О каком объекте идет речь, по какому адресу, кто собственник, кто заказчик, какие работы предусматриваются, какой вывод экспертизы: {text[:2000]}"
            logger.info(f"Длина текста для GPT: {len(text4gpt)}")
        
        logger.info('Запрос к GPT...')    
        txt = ''
        with model.chat_session():
            g = model.generate(text4gpt, max_tokens=256, streaming=True)
            for i, v in enumerate(g):
                print(v, end="", flush=True)
                txt += v
        
        logger.info('Ответ от GPT получен')
        return txt
    except Exception as e:
        logger.error(f"Ошибка в функции get_gpt_text: {str(e)}")
        return "Ошибка при обработке текста ИИ"

###
### Построение карты
###


def get_kadastr(text):
    try:
        os.makedirs("maps", exist_ok=True)
        
        # Извлекаем кадастровые номера из текста
        kadastr = re.findall(r'\d{1,2}:\d{1,2}:\d{1,7}:\d{1,7}', text)
        kadastr = list(set(kadastr))  # Удаляем дубликаты
        
        logger.info(f"Найдены кадастровые номера: {kadastr}")
        
        if not kadastr:
            logger.info("Кадастровые номера не найдены")
            return None

        gdf_merged = gpd.GeoDataFrame()

        for kn in kadastr:
            try:
                logger.info(f"Обработка кадастрового номера: {kn}")
                
                # Получаем данные через rosreestr2coord
                area = Area(kn, with_proxy=True)
                coords = area.to_geojson_poly()
                coords=json.loads(coords)
                #Проверяем, идет ли речь о точке или полигоне
                if isinstance(coords['geometry']['coordinates'][0], list):
                    coords=coords['geometry']['coordinates'][0]
                    polygon_geom = Polygon(coords)
                else: 
                    coords=coords['geometry']['coordinates']
                    coords[0], coords[1] = coords[0]/100000, coords[1]/100000
                    polygon_geom = Point(coords)

                
                # Создаем GeoDataFrame из полигона
                
                    
                gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) 
                gdf_merged = pd.concat([gdf_merged, gdf], ignore_index=True)
                
            except Exception as e:
                logger.error(f"Ошибка обработки кадастрового номера {kn}: {str(e)}")
                continue

        if gdf_merged.empty:
            logger.info("Не удалось получить геоданные по кадастровым номерам")
            return None
            
        # Создаем карту
        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_merged.plot(ax=ax, alpha=0.5, edgecolor='red')
            
            # Добавляем подложку карты
            try:
                cx.add_basemap(ax, crs=gdf_merged.crs.to_string(), 
                             source=cx.providers.OpenStreetMap.Mapnik, zoom=18)
            except Exception as map_error:
                logger.warning(f"Ошибка загрузки подложки карты: {str(map_error)}")
                # Рисуем без подложки, если не удалось загрузить
            
            filename = f"{uuid.uuid4()}.png"
            plt.savefig(f"maps/{filename}", bbox_inches='tight', dpi=100)
            plt.close()
            
            logger.info(f"Карта сохранена как: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Ошибка при создании карты: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Критическая ошибка в функции get_kadastr: {str(e)}")
        return None
    
###
### Извлечение изображений из PDF и создание миниатюр
###
def extract_images_and_create_thumbnail(pdf_path):
    try:
        # Создаем папку для миниатюр, если ее нет
        os.makedirs("thumbnails", exist_ok=True)
        
        # Конвертируем PDF в изображения
        images = convert_from_path(pdf_path, first_page=1, last_page=10, fmt='jpeg')
        
        if not images:
            logger.info("Не удалось извлечь изображения из PDF")
            return None
            
        # Отбираем только большие изображения (ширина или высота > 500px)
        large_images = []
        for img in images:
            if img.width > 500 or img.height > 500:
                large_images.append(img)
        
        if not large_images:
            logger.info("Нет подходящих изображений для миниатюр")
            return None
            
        # Ограничиваем количество изображений до 25 (для сетки 5x5)
        large_images = large_images[:25]
        
        # Определяем размер миниатюр (100x100 пикселей)
        thumb_size = 100
        cols = 5
        rows = min(5, (len(large_images) + cols - 1)) // cols
        
        # Создаем холст для миниатюр
        thumbnail = Image.new('RGB', (cols * thumb_size, rows * thumb_size))
        
        # Размещаем миниатюры на холсте
        for i, img in enumerate(large_images):
            if i >= cols * rows:
                break
                
            # Создаем миниатюру
            img.thumbnail((thumb_size, thumb_size))
            
            # Вычисляем позицию для вставки
            x = (i % cols) * thumb_size
            y = (i // cols) * thumb_size
            
            # Вставляем миниатюру на холст
            thumbnail.paste(img, (x, y))
        
        # Сохраняем миниатюру
        thumb_filename = f"{uuid.uuid4()}.jpg"
        thumbnail.save(f"thumbnails/{thumb_filename}")
        
        logger.info(f"Миниатюра сохранена как: {thumb_filename}")
        return thumb_filename
    except Exception as e:
        logger.error(f"Ошибка при создании миниатюр: {str(e)}")
        return None

###
### Код загрузки ГИКЭ
###
def getgike(link, headers, short_link=''):
    try:
        with requests.get(link, headers=headers, stream=True, timeout=10, verify=False) as pdf_bytes:
            pdf_bytes.raise_for_status()
            with open('temp.pdf', 'wb') as p:
                pbar = tqdm(total=int(pdf_bytes.headers['Content-Length']))
                for chunk in pdf_bytes.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        p.write(chunk)
                        pbar.update(len(chunk))
                p.seek(0, os.SEEK_END)
                
                #Распознание PDF
                try:
                    ocrmypdf.ocr('temp.pdf', 'tempocr.pdf', l='rus', pages='1-50', output_type='pdf', optimize=0) 
                    docpath = 'tempocr.pdf' #имя файла, из которого будем извлекать текст
                    logger.info("PDF успешно распознан")
                except (ocrmypdf.exceptions.PriorOcrFoundError, ocrmypdf.exceptions.TaggedPDFError) as e:
                    logger.info("OCR не требуется.")
                    docpath = 'temp.pdf'
                except Exception as e: 
                    logger.error(f"Ошибка в модуле распознавания: {str(e)}")
                    docpath = 'temp.pdf'
                
                # Извлекаем изображения и создаем миниатюры
                thumb_filename = extract_images_and_create_thumbnail(docpath)
                
                read_pdf = PdfReader(docpath)
                count = len(read_pdf.pages)
                pages_txt = ''

                for x in range(count)[:200]:
                    page = read_pdf.pages[x]    
                    try:
                        page_text = page.extract_text()
                    except Exception as e:
                        logger.warning(f"Ошибка извлечения текста со страницы {x}: {str(e)}")
                        page_text = ''
                    pages_txt = pages_txt + page_text
        
        logger.info(f"Текст из PDF извлечен, длина: {len(pages_txt)} символов")
        return pages_txt, thumb_filename
    except Exception as e:
        logger.error(f"Ошибка в функции getgike: {str(e)}")
        return "", None


if __name__ == "__main__":

    logger = setup_logging()

    keywords = ["предусматривает", "проектом", "собственник", "заказчик", "вывод"]

    logger.info('Загрузка модели')

    # Путь к модели также берётся из файла user_data
    try:
        model = GPT4All(user_data.model_path, allow_download=False)
        logger.info('Загрузка модели выполнена')
    except Exception as e:
        logger.error(f'Ошибка загрузки модели: {str(e)}')
        raise
        
    # Создаем базу данных и таблицу, если их нет
    try:
        with sqlite3.connect("kgiop_db.db") as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS kgiop (
                    date TEXT,
                    link TEXT,
                    summary TEXT,
                    text TEXT,
                    map_filename TEXT,
                    thumb_filename TEXT
                );
            """)
        logger.info("База данных инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {str(e)}")
        raise

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0'
    }
    
    logger.info('Запрашиваем страницу экспертиз КГИОП')
        
    # Получаем нынешний год для подстановки в ссылку
    dt = datetime.datetime.now()
    year_only = dt.year
        
    try:
        response = requests.get(
            f'https://kgiop.gov.spb.ru/deyatelnost/zaklyucheniya-gosudarstvennyh-istoriko-kulturnyh-ekspertiz/gosudarstvennye-istoriko-kulturnye-ekspertizy-za-{year_only}-g/',
            headers=headers,
            verify=False
        )
        response.raise_for_status()
        response = response.text
        soup = BeautifulSoup(response, "lxml")
        eventtypesall = soup.find_all('a')
        logger.info(f"Список из {len(eventtypesall)} ссылок получен. Начинаем загрузку экспертиз")
        sleep(1)

        index = 0
        for i in eventtypesall: 
            link_capt = '' # Титул экспертизы     
            if "Срок рассмотрения обращений" in i.text:
                link_capt = 'Экспертиза ' + i.find_parent('td').find_previous_sibling('td').text + ' (часть составной экспертизы)'    
            if 'disk.yandex.ru' in i['href']: # если файл выложен на яндекс-диск
                index += 1
                if index > 19:
                    logger.info("Достигнут лимит в 20 экспертиз за один запуск")
                    break
                logger.info(f'Загружаем экспертизу № {index}')
                logger.info(i.text)
                apilink = f'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={i["href"]}'
                response_json = requests.get(apilink).json()
                link = response_json.get("href")
                
                # Есть ли уже в архиве этот документ?
                if indb(i['href']):
                    logger.info(f"Документ по ссылке {link} уже есть в архиве, пропускаем его")
                    continue
            
                logger.info(link)
                link_caption = i.text

                text, thumb_filename = getgike(link, headers)
                summary = extract_text_fragments(text, keywords)
                text4gpt = text[:1000] + summary[:500] + summary[-500:]
                text_result = get_gpt_text(text4gpt)
                logger.info(text_result)
                
                map_filename = get_kadastr(text)
                logger.info(f"Карта сохранена как: {map_filename}")
                logger.info(f"Миниатюры сохранены как: {thumb_filename}")
            
                todb(i['href'], text_result, text, map_filename, thumb_filename)
                    
                logger.info('Ожидание следующего запроса к сайту')
                sleep(randint(5,10))
                
            elif '/media/uploads/userfiles/' in i['href']: # если файл выложен на сайте кгиоп
                index += 1
                if index > 19:
                    logger.info("Достигнут лимит в 20 экспертиз за один запуск")
                    break
                logger.info(f'Загружаем экспертизу № {index}')
                link = "https://kgiop.gov.spb.ru" + i['href']
            
                # Есть ли уже в архиве этот документ?
                if indb(link): 
                    logger.info(f"Документ по ссылке {link} уже есть в архиве, пропускаем его")
                    continue

                text, thumb_filename = getgike(link, headers=headers)
                summary = extract_text_fragments(text, keywords)
                text4gpt = text[:1000] + summary[:500] + summary[-500:]
                text_result = get_gpt_text(text4gpt)
                logger.info(text_result)

                map_filename = get_kadastr(text)
                logger.info(f"Карта сохранена как: {map_filename}")
                logger.info(f"Миниатюры сохранены как: {thumb_filename}")
                
                todb(link, text_result, text, map_filename, thumb_filename)
                if link_capt == '': 
                    link_capt = 'Экспертиза ' + i.text

                logger.info('Ожидание следующего запроса к сайту')
                sleep(randint(2,4))
                
    except Exception as e:
        logger.error(f"Критическая ошибка в основном цикле: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    logger.info("Программа завершена успешно")


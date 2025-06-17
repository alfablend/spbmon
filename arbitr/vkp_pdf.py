#Модуль выгрузки и распознания pdf

import requests
import os
from pypdf import PdfReader
from tqdm import tqdm

import traceback
import logging
import re

def getpdf (link, cookies, headers, extract_images=False):
            
    with requests.get(link, cookies=cookies, headers=headers, stream=True, timeout=10) as pdf_bytes:
        pdf_bytes.raise_for_status()
        with open('temp.pdf', 'wb') as p:
            pbar = tqdm(total=int(pdf_bytes.headers['Content-Length']))
            for chunk in pdf_bytes.iter_content(chunk_size=8192):
             
                if chunk:  
                    p.write(chunk)
                    pbar.update(len(chunk))
            p.seek(0, os.SEEK_END)
            
            docpath='temp.pdf'
            #Распознание PDF

            read_pdf = PdfReader(docpath)
            count = len(read_pdf.pages)
            pages_txt = ''

            # Извлекаем постранично текст
            found_page=0 # Сюда будем сохранять страницу, на которой найдено слово "распоряжение"
            for x in range(count):
                page = read_pdf.pages[x]    
                page_text = page.extract_text()
                pages_txt = pages_txt + page_text
            
    return pages_txt
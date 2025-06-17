# Автоматический пересказ решений арбитражных судов,
# Требуется аккаунт "Контур.Фокуса" для работы

from vkp_pdf import getpdf # Обработка PDF-файлов
import vkp_db as db # Логика работы с базой данных
import user_data # Заголовки запроса, включая cookies

from gpt4all import GPT4All #Поддержка ИИ

import requests

from bs4 import BeautifulSoup  

from time import sleep
from tqdm import tqdm

#Заголовки запроса, включая cookie, подгружаются из файла
#user_data.py (нет в репозитории, нужно создать и вставить туда словарь
#headers с нужными переменными (можно взять через отладчик браузера
#и curlconverter.com)). Отдельные cookies не нужны.
headers, cookies = user_data.headers, {}

print('Загрузка модели')

# Путь к модели также берётся из файла user_data
model = GPT4All(user_data.model_path, allow_download=False)
print('Загрузка выполнена')

response = requests.get('https://focus.kontur.ru/content/mon', headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.content, "lxml")
eventtypesall = soup.find_all('a', {'class': 'hover-underline org-changes-document'})
sleep(1)

for i in tqdm(eventtypesall):

    link="https://focus.kontur.ru" + i['href']
    
    #Проверяем, был ли уже загружен документ в архив (тогда пропускаем его, возвращаемся в начало цикла)
    if db.indb(link): 
        print ("Документ по ссылке " + link + "уже есть в архиве, пропускаем его")
        continue
    text = getpdf(link, cookies, headers)
    if not 'Санкт-Петербург' in text:
        print (f'Документ по ссылке {link} не касается Петербурга, пропускаем его')
        continue
    if not any(substring in text for substring in ['ешение', 'остановление', 'пределение']):
        print (f'Документ по ссылке {link} не является решением, пропускаем его')
        continue
    #print(text[:1000])
    if len(text)>2000:
        text4gpt=f"Перескажи коротко текст c указанием истца, ответчика, сути дела и его итогов: {text[:1000]} {text[-1000:]}"
        print(len(text4gpt))
    else:
        text4gpt=f"Перескажи коротко текст c указанием истца, ответчика, сути дела и его итогов: {text[:2000]}"
        print(len(text4gpt))
    print('Ждем GPT')    
    ln=0
    with model.chat_session():
        g=model.generate(text4gpt, max_tokens=256, streaming=True)
        txt=''
        for i, v in enumerate(g):
            print(v, end = "", flush = True)
            ln=i
            txt+=v
    print('Длина ответа: ', ln)        
    db.todb('arbitr', link, txt)
    img=''
    link=''
    #Отправка в телеграм отключена
    #vkp_app.vkp_telegram.to_telegram('arbitr', img, link, 'Арбитражное дело', data4telegram)

    sleep(1)  

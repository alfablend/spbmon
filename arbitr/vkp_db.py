#Модуль работы с базой данных

import pandas as pd
import datetime
import shutil
import uuid

import cv2

import traceback
import logging

#Чтение всех CSV из папки и их объединение
#https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe

import glob
import os

import numpy as np

df_dict={}

def load_db():
    global df_dict
    global df_merged
    path=''
    all_files = glob.glob(os.path.join(path, "*.csv"))
        
    df_dict={}
    for f in all_files:
        
        #чистим ключи от относительного пути и расширения файла
        key=f.split('\\')[-1]
        key=key.split('.csv')[0]
        try:
            df_dict[key]=pd.read_csv(f, header=0, on_bad_lines="skip", engine='python')
        except:
            print(f'Не удалось прочитать файл {f}')
    try:
        df_merged = pd.concat(list(df_dict.values()), ignore_index=True)
    except:
        print('Проблема чтения базы данных, создаем пустую базу')
        df_merged=pd.DataFrame(columns=['type', 'date', 'link', 'text'])
        
        
def indb (link):
    #В режиме отладки сообщаем, что ссылки нет в базе
    
    global df_merged
    if link in df_merged['link'].values: return True
    else: return False
 
def todb (type_, link, text, img=''):
    
    global df_merged
    global df_dict
    unique_filename=''
    if isinstance(img, np.ndarray):
        unique_filename = str(uuid.uuid4())+'.jpg'
        path_to = 'vkp_app\\database\\' + type_ + '_files' + '\\' + unique_filename
        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except: 
            print ('Проблема записи картинки в базу!')
            pass
        cv2.imwrite(path_to, img, compression_params)
        
    #Есть ли уже этот монитор в базе?
    if type_ in df_dict:    
        df_dict[type_]=df_dict[type_]._append({'type':type_, 'date':str(datetime.datetime.now()), 'link':link, 'text':text, 'unique_filename':unique_filename}, ignore_index=True)
        
        #Чистим базу от ненужного
        #Удаляем картинки ненужных записей после 500
                
        df_dict[type_] =  df_dict[type_].iloc[-500:] #Чтобы база не разрасталась, оставляем в ней только 500 последних значений
        
        
    else:
        df_dict[type_] = pd.DataFrame ({'type':type_, 'date':str(datetime.datetime.now()), 'link':link, 'text':text, 'unique_filename':unique_filename}, index=[0])
    
    output_path=type_ + '.csv'
    df_dict[type_].to_csv(output_path, index=False)
    df_merged = pd.concat(list(df_dict.values()), ignore_index=True)
    
    return(unique_filename)
    
load_db()
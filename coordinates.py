import requests
import pandas as pd
import json
from tqdm import tqdm 

def yandex_geocode(address, api_key, inn):
    address = address.replace(' ', '+').replace('. ', '+')
    url = f'https://geocode-maps.yandex.ru/v1/?apikey={api_key}&geocode={address}&format=json' 
    response = requests.get(url).json()
    try:
        pos = response["response"]["GeoObjectCollection"]["featureMember"][0]["GeoObject"]["Point"]["pos"]
        lon, lat = map(float, pos.split())  
        return {"inn": inn, "address": address, "lat": lat, "lon": lon, "status": "success"}
    
    except Exception as e:
        return {"inn": inn, "address": address, "lat": None, "lon": None, "status": f"error: {str(e)}"}


def process_supplier(input_file, output_file, api_key):
    df = pd.read_excel(input_file) 
    print(df)
    addresses = df['Адрес'].tolist()
    inn = df['ИНН поставщика'].tolist()

    res = []
    i = 0
    for i in range(len(addresses)):
        result = yandex_geocode(addresses[i], api_key, inn[i])
        res.append(result)
        print(f'{inn[i]}: done')
    
    result_df = pd.DataFrame(res)
    result_df.to_csv(output_file, index=False, encoding="utf-8")


API_KEY = "916761b8-2963-4217-a148-e9c603fe3b10" 

for i in range(1, 6):
    print(i)
    INPUT_FILE = f"assemble/data_tend/tend{i}.xlsx"  
    OUTPUT_FILE = f"tender{i}_res_all.csv"
    process_supplier(INPUT_FILE, OUTPUT_FILE, API_KEY)

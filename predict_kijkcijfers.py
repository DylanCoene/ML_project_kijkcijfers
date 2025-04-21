# Structuur nieuwe voorspellingen maken:
# 1. Verkrijg de kijkcijfer (KC) data van laatste 2-3 weken (EXCLUSIEF DE GEWESTE PREDICTIES)
# 2. Clean KC data
# 3. Haal de weerdata op van minDate tot maxDate van de KC data
# 4. Merge weerdata en KC data
# 5. Drop missing values
# 6. Feature engineering
#    1. Timestamp
#    2. Lag features (haal deze uit de opgehaalde data)
#    3. Target Encoding (haal deze uit opgehaalde data)
# 7. Gebruik model om te predicten ADHV de data

from datetime import datetime, timedelta
import requests
import time
import pandas as pd
import holidays
from category_encoders import TargetEncoder
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb
import numpy as np
import warnings
warnings.simplefilter("ignore")

# Fetch Kijkcijferdata
def fetch_kijkcijfers(start_date, end_date):
    data_list = []
    print(f"Start fetching kijkcijfers van {start_date.date()} tot {end_date.date()}")

    # Loop door elke dag
    current_date = start_date
    while current_date <= end_date:
        datum = f"{current_date.year}-{current_date.month}-{current_date.day}"
        url = f"https://api.cim.be/api/cim_tv_public_results_daily_views?dateDiff={datum}&reportType=north"
        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                programma_lijst = data.get('hydra:member', [])
                
                for programma in programma_lijst:
                    try:
                        data_list.append({
                            'dateDiff': programma.get('dateDiff'),
                            'ranking': programma.get('ranking'),
                            'description': programma.get('description'),
                            'channel': programma.get('channel'),
                            'startTime': programma.get('startTime'),
                            'rLength': programma.get('rLength'),
                            'rateInK': programma.get('rateInK'),
                            'live': programma.get('live')
                        })
                        
                    except Exception as e:
                        print(f"Fout bij verwerken programma op {datum}: {e}")
            else:
                print(f"Geen data voor {datum} (HTTP {response.status_code})")
                
        except Exception as e:
            print(f"Fout bij ophalen {datum}: {e}")
        
        current_date += timedelta(days=1)
    
    print(f"Einde fetching kijkcijfers")
    # Maak een dataframe van de data
    df = pd.DataFrame(data_list)
    return df

# Fetch weerdata
def fetch_weerdata(start_date, end_date):
    # Locatie voor Vlaanderen (Brussel als centraal punt)
    latitude = 50.8503
    longitude = 4.3517

    # API endpoints voor historische data en forecast
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    # Relevante hourly variabelen voor het ML model
    hourly_vars = [
        "temperature_2m",   # Gemiddelde temperatuur per uur
        "weathercode",      # Weertype als code per uur
        "precipitation",    # Totale neerslag per uur
        "rain",             # Regen per uur
        "snowfall",         # Sneeuwval per uur
        "cloudcover",       # Bewolking per uur
        "windspeed_10m"     # Windsnelheid per uur
    ]
    
    # Gemeenschappelijke API parameters
    common_params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": hourly_vars,
        "timezone": "Europe/Brussels",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
        "windspeed_unit": "kmh"
    }
    
    today = datetime.today().date()
    dataframes = []
    
    # Historische data (als start_date < vandaag)
    if start_date.date() < today:
        # Bepaal de einddatum voor historische data (niet later dan gisteren)
        hist_end_date = min(end_date.date(), today - timedelta(days=1))
        params_hist = common_params.copy()
        params_hist.update({
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": hist_end_date.strftime('%Y-%m-%d')
        })
        print(f"Fetching historische weerdata van {params_hist['start_date']} tot {params_hist['end_date']}")
        response = requests.get(archive_url, params=params_hist)
        if response.status_code == 200:
            data = response.json().get("hourly", {})
            df_hist = pd.DataFrame({
                "time": data.get("time", []),
                "temperature": data.get("temperature_2m", []),
                "weather_code": data.get("weathercode", []),
                "precipitation": data.get("precipitation", []),
                "rain": data.get("rain", []),
                "snowfall": data.get("snowfall", []),
                "cloudcover": data.get("cloudcover", []),
                "windspeed": data.get("windspeed_10m", [])
            })
            if not df_hist.empty:
                df_hist['time'] = pd.to_datetime(df_hist['time'])
                df_hist['hour'] = df_hist['time'].dt.hour              # Uur van de dag
                df_hist['day_of_week'] = df_hist['time'].dt.dayofweek      # 0 = maandag, 6 = zondag
                df_hist['month'] = df_hist['time'].dt.month                # Maand (1 t/m 12)
                df_hist['year'] = df_hist['time'].dt.year                  # Jaar
                dataframes.append(df_hist)
        else:
            print(f"Fout bij het ophalen van historische data: {response.status_code}")
            print(response.text)
    
    # Forecast data (als end_date >= vandaag)
    if end_date.date() >= today:
        # Voor forecast gebruiken we data vanaf vandaag (of start_date als deze later is dan vandaag)
        forecast_start = max(start_date, datetime.combine(today, datetime.min.time()))
        params_forecast = common_params.copy()
        params_forecast.update({
            "start_date": forecast_start.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d')
        })
        print(f"Fetching forecast weerdata van {params_forecast['start_date']} tot {params_forecast['end_date']}")
        response = requests.get(forecast_url, params=params_forecast)
        if response.status_code == 200:
            data = response.json().get("hourly", {})
            df_forecast = pd.DataFrame({
                "time": data.get("time", []),
                "temperature": data.get("temperature_2m", []),
                "weather_code": data.get("weathercode", []),
                "precipitation": data.get("precipitation", []),
                "rain": data.get("rain", []),
                "snowfall": data.get("snowfall", []),
                "cloudcover": data.get("cloudcover", []),
                "windspeed": data.get("windspeed_10m", [])
            })
            if not df_forecast.empty:
                df_forecast['time'] = pd.to_datetime(df_forecast['time'])
                df_forecast['hour'] = df_forecast['time'].dt.hour
                df_forecast['day_of_week'] = df_forecast['time'].dt.dayofweek
                df_forecast['month'] = df_forecast['time'].dt.month
                df_forecast['year'] = df_forecast['time'].dt.year
                dataframes.append(df_forecast)
        else:
            print(f"Fout bij het ophalen van forecast data: {response.status_code}")
            print(response.text)
    
    if dataframes:
        # Combineer en sorteer de dataframes op tijd
        df = pd.concat(dataframes).sort_values("time").reset_index(drop=True)
    else:
        df = pd.DataFrame()
    
    print("Einde fetching weerdata")
    return df

# Clean data
def clean_KC_data(df):
    # Verwijder de kolom 'ranking' als deze aanwezig is, deze is nutteloos
    if 'ranking' in df.columns:
        df.drop('ranking', axis=1, inplace=True)

    # Verwijder rijen met null waarden, deze zijn fout geregistreerd in de DB
    df.dropna(inplace=True)

    # Verwijder rijen waarvan tijdformaat nie overeenkomt met xx:xx:xx
    time_pattern = r'^\d{1,2}:\d{1,2}:\d{1,2}$'

    df = df[
        df["startTime"].str.match(time_pattern, na=False) & 
        df["rLength"].str.match(time_pattern, na=False)
    ].copy()

    # rLength omzetten naar seconden -> duration_sec
    # dateDiff omzetten naar datetime

    df['duration_sec'] = pd.to_timedelta(df['rLength']).dt.total_seconds()
    df['date'] = pd.to_datetime(df["dateDiff"]).dt.date

    # Enkel viewers cleanen als het niet to predict is
    if 'rateInK' in df.columns:
        # rateInK omzetten naar int -> viewers
        df['viewers'] = df['rateInK'].apply(lambda x: int(''.join(str(x).split('.'))))

    #startTime: 24:30:00 omzetten naar 00:30:00 en dateDiff 1 dag verhogen
    def fix_next_day(rij):
        time_parts = rij['startTime'].split(':')
        if int(time_parts[0]) >= 24:
            time_parts[0] = str(int(time_parts[0]) - 24).zfill(2)
            rij['date'] += timedelta(days=1)
        rij['startTime'] = ':'.join(time_parts)
        return rij

    df = df.apply(fix_next_day, axis=1)

    # timestamp kolom toevoegen op basis van dateDiff en startTime
    df['timestamp'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['startTime'].astype(str))

    # Live: boolean maken, of getal houden?? 28 > 7 > 1 > 0 heeft groter getal ook correlatie met kijkcijfer??? 
    # Beslissing: Live kolom laten als int

    # Hour year month day toevoegen
    df['hour'] = pd.to_datetime(df['startTime']).dt.hour
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day'] = pd.to_datetime(df['date']).dt.day

    # Onnodige kolommen verwijderen
    columns_to_drop = ['startTime', 'rLength']
    if 'rateInK' in df.columns:
        columns_to_drop.append('rateInK')
    df.drop(columns_to_drop, axis=1, inplace=True)

    if 'viewers' in df.columns:
        df = df[['timestamp', 'date', 'year', 'month', 'day', 'hour', 'channel', 'description', 'duration_sec', 'live', 'viewers']]
    else:
        df = df[['timestamp', 'date', 'year', 'month', 'day', 'hour', 'channel', 'description', 'duration_sec', 'live']]

    df.rename(columns={'description': 'program'}, inplace=True)

    return df

def clean_weer_data(weer_data):
    weer_data['datetime'] = pd.to_datetime(weer_data['time'])
    weer_data['date'] = weer_data['datetime'].dt.date
    weer_data['hour'] = weer_data['datetime'].dt.hour
    weer_data['year'] = weer_data['datetime'].dt.year
    weer_data['month'] = weer_data['datetime'].dt.month
    weer_data['day'] = weer_data['datetime'].dt.day

    weer_data = weer_data.drop(columns=['time', 'day_of_week'])
    return weer_data

def one_hot_encode_features(df):
    # One hot encoding voor 'live', 'channel', 'weather_code', 'season'
    with open('./models/one_hot_encoder.pkl', 'rb') as file:
        one_hot_encoder = pickle.load(file)
    
    df_cat = df[['live', 'channel', 'weather_code', 'season']]

    df_1hot = one_hot_encoder.transform(df_cat)

    one_hot_output = pd.DataFrame(df_1hot.toarray(), 
                                  columns=one_hot_encoder.get_feature_names_out(), 
                                  index=df_cat.index)

    # Voeg de one-hot encoded data toe en drop de originele kolommen
    df = df.drop(columns=['live', 'channel', 'weather_code', 'season'])
    df = pd.concat([df, one_hot_output], axis=1)
    
    return df

# Feature engineering

# Functie om seizoen uit datum te halen
def get_season(date):
    if date.month in [3, 4, 5]:
        return 'lente'
    elif date.month in [6, 7, 8]:
        return 'zomer'
    elif date.month in [9, 10, 11]:
        return 'herfst'
    else:
        return 'winter'

def timestamp_feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['season'] = df['timestamp'].apply(get_season)

    # weekday toevoegen
    df['weekday'] = df['timestamp'].dt.weekday

    # uur toevoegen
    df['hour'] = df['timestamp'].dt.hour

    # dag toevoegen
    df['day'] = df['timestamp'].dt.day

    # maand toevoegen
    df['month'] = df['timestamp'].dt.month

    # isWeekend toevoegen
    df['isWeekend'] = df['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)

    # isPrimeTime toevoegen
    df['isPrimeTime'] = df['hour'].apply(lambda x: 1 if x >= 18 and x <= 23 else 0)

    # isHoliday toevoegen
    be_holidays = holidays.BE()
    df['isHoliday'] = df['timestamp'].apply(lambda x: 1 if x in be_holidays else 0)

def create_lag_features(pred_hist_df, historische_KC_weer_data):
    # to predict samenvoegen met historische data om lag features te maken
    pred_hist_df['viewers'] = None
    pred_hist_df = pd.concat([historische_KC_weer_data, pred_hist_df], ignore_index=True)

    # Lag features maken voor viewers
    for i in range(1, 4):
        pred_hist_df[f'viewers_lag_{i}'] = pred_hist_df.sort_values('timestamp').groupby(['program'])['viewers'].shift(i)
        pred_hist_df[f'viewers_lag_{i}'] = pred_hist_df[f'viewers_lag_{i}'].fillna(pred_hist_df.groupby(['program'])['viewers'].transform('mean'))
        
    return pred_hist_df

# Pipeline

def preprocess(df):
    # Verander de kolomnamen naar de juiste namen
    df.rename(columns={
        'Programma': 'description',
        'Zender': 'channel',
        'Datum': 'dateDiff',
        'Start': 'startTime',
        'Duur': 'rLength',
        # Typo in gegeven csv
        'Datum ': 'dateDiff',
        'Start ': 'startTime',
    }, inplace=True)

    # Voeg live kolom toe om te laten werken in de pipeline
    df['live'] = 0

    df['dateDiff'] = pd.to_datetime(df['dateDiff'])

    end_date = df['dateDiff'].max()
    start_date = end_date - timedelta(weeks=3)

    # Haal historische data op
    historische_kijkcijfers = fetch_kijkcijfers(start_date, end_date - timedelta(days=1))
    historische_kijkcijfers_clean = clean_KC_data(historische_kijkcijfers)
    historische_weerdata = fetch_weerdata(start_date, end_date)
    historische_weerdata_clean = clean_weer_data(historische_weerdata)

    # Merge historische kijkcijfers en weerdata
    kijkcijfers_weer = pd.merge(historische_kijkcijfers_clean, historische_weerdata_clean, on=['date', 'hour'], how='left')
    historische_KC_weer_data = kijkcijfers_weer[['timestamp', 'channel', 'program', 'duration_sec', 'live', 'viewers', 'weather_code', 'temperature', 'rain', 'windspeed', 'snowfall', 'precipitation']]

    # Merge to predict data met weerdata
    to_predict = df.copy()
    to_predict_clean = clean_KC_data(to_predict)
    to_predict_data = pd.merge(to_predict_clean, historische_weerdata_clean, on=['date', 'hour'], how='left')

    # Drop rows met null waarden
    historische_KC_weer_data.dropna(inplace=True)

    # Feature engineering op timestamp
    timestamp_feature_engineering(to_predict_data)
    timestamp_feature_engineering(historische_KC_weer_data)

    # Gebruik de functie in de make_prediction functie
    pred_hist_df = create_lag_features(to_predict_data, historische_KC_weer_data)
    
    # Haal de to_predict data er terug uit
    to_predict_data = pred_hist_df[pred_hist_df['viewers'].isnull()]

    # One hot encoding voor programma
    to_predict_data = one_hot_encode_features(to_predict_data)

    # Target encoding voor programma
    with open('./models/target_encoder.pkl', 'rb') as file:
        target_encoder = pickle.load(file)
    
    to_predict_data['program'] = target_encoder.transform(to_predict_data['program'])

    to_predict_data.drop(columns=['date', 'cloudcover', 'viewers', 'year_x', 'month_x', 'day_x', 'year_y', 'month_y', 'day_y', 'datetime'], inplace=True)

    # Selecteer enkel numerieke kolommen
    to_predict_data_num = to_predict_data.select_dtypes(include=[np.number])

    return to_predict_data_num

def make_predictions(data):
    # Update the pipeline to use the FunctionTransformer
    preprocess_pipeline = Pipeline([
        ('preprocess', FunctionTransformer(preprocess)),
        ('std_scaler', StandardScaler())
    ])

    preprocessed_data = preprocess_pipeline.fit_transform(data)

    with open('./models/tuned_xgb_model.pkl', 'rb') as file:
        xgb_model = pickle.load(file)

    predictions = xgb_model.predict(preprocessed_data)
    
    return predictions


if __name__ == "__main__":
    to_predict = pd.read_csv('./to_predict/test_input.csv', sep=';')

    preds = make_predictions(to_predict)
    to_predict['predictions'] = preds
    print("Voorspellingen:")
    print(to_predict)
    to_predict.to_csv('./to_predict/predictions.csv', sep=';', index=False)
    print("Voorspellingen opgeslagen in predictions.csv")

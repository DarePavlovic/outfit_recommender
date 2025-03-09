import openmeteo_requests
from twilio.rest import Client
import joblib
import requests_cache
import pandas as pd
from retry_requests import retry
import schedule
import time
from dotenv import load_dotenv
import os

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)
load_dotenv()
# Configuration
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER') # Twilio phone number
user_phone_number = os.getenv('USER_PHONE_NUMBER') # Your phone number
lat = 41.40423
lon = 2.17301

model = joblib.load('outfit_recommender_model.joblib')

def get_weather(lat, lon):
    #url = f'https://api.openweathermap.org#current.json?key={api_key}&q={location}'
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min", "uv_index_max", "uv_index_clear_sky_max", "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max"],
	    "timezone": "Europe/Berlin",
	    "forecast_days": 1
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    daily = response.Daily()
    weather_data = {
        'Maximum Temperature': daily.Variables(0).ValuesAsNumpy()[0],
        'Minimum Temperature': daily.Variables(1).ValuesAsNumpy()[0],
        'Maximum Apparent Temperature': daily.Variables(2).ValuesAsNumpy()[0],
        'Minimum Apparent Temperature': daily.Variables(3).ValuesAsNumpy()[0],
        'Precipitation Sum': daily.Variables(6).ValuesAsNumpy()[0],
        'Rain Sum': daily.Variables(7).ValuesAsNumpy()[0],
        'Showers Sum': daily.Variables(8).ValuesAsNumpy()[0],
        'Snowfall Sum': daily.Variables(9).ValuesAsNumpy()[0],
        'Precipitation Hours': daily.Variables(10).ValuesAsNumpy()[0],
        'Maximum Wind Speed': daily.Variables(11).ValuesAsNumpy()[0],
        'Maximum Wind Gusts': daily.Variables(12).ValuesAsNumpy()[0],
        'UV Index': daily.Variables(4).ValuesAsNumpy()[0],
        'UV Index Clear Sky': daily.Variables(5).ValuesAsNumpy()[0]
    }
    return weather_data

def recommend_outfit(weather):
    weather_features = pd.DataFrame([weather])

    # Predict the outfit using the trained model
    outfit = model.predict(weather_features)[0]

    return outfit

def send_whatsapp_message(twilio_account_sid, twilio_auth_token, twilio_phone_number, user_phone_number, message):

    account_sid = twilio_account_sid
    auth_token = twilio_auth_token
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_='whatsapp:'+twilio_phone_number,
        body='Today\'s outfit recommendation: ' + message,
        to='whatsapp:'+user_phone_number
    )   

def job():
    try:
        weather = get_weather(lat, lon)
        outfit_recommendation = recommend_outfit(weather)
        print (outfit_recommendation)
        send_whatsapp_message(twilio_account_sid, twilio_auth_token, twilio_phone_number, user_phone_number, outfit_recommendation)
    except KeyError as e:
        print(f'Error: {e}')

def main():
     # Schedule the job every day at 8 AM
    schedule.every().day.at("13:22").do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    main()
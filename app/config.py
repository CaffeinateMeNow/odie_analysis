from dotenv import load_dotenv
import os

load_dotenv()
ELASTICSEARCH_HOST = os.getenv('ELASTICSEARCH_HOST')
BACKEND_HOST = os.getenv('BACKEND_HOST')
BACKEND_EMAIL = os.getenv('BACKEND_EMAIL')
BACKEND_PASSWORD = os.getenv('BACKEND_PASSWORD')

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("IBM_WATSONX_API_KEY")
project_id = os.getenv("IBM_WATSONX_PROJECT_ID")

file_name = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

host_url = "https://jp-tok.ml.cloud.ibm.com"

model_id = 'meta-llama/llama-3-1-70b-instruct'

# print(api_key)
# print(project_id)
# print(file_name)
# print(url)
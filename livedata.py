import requests

url = "https://railradar.in/api/v1/trains/live-map"

payload = {}
headers = {
  'x-api-key': 'rri_eyJleHAiOjE3NTc4NzUyODEzMzEsImlhdCI6MTc1Nzc4ODg4MTMzMSwidHlwZSI6ImludGVybmFsIiwicm5kIjoiVkJrYWpCUERINlY5In0=_NDQyYjk0MzljNWZkNWEyM2Y4MTliMzQ4MjczNDUyNmFkNTZlM2NhMTg0ZTliOTQ2NWEzYTY4ZDc0MWU5ODdmMg==',
  'Cookie': 'user_id=ca319a54450a46e7929a2b79a31592c8'
}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

multipart_data = MultipartEncoder(
    fields={
            # a file upload field
            'file': ('videoplayback.wav', open('videoplayback.wav', 'rb'))
           }
    )
    
response = requests.post('http://localhost:5000', data=multipart_data,
                  headers={'Content-Type': multipart_data.content_type})

with open("a.wav", "wb") as f:
    f.write(response.content)
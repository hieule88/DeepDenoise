import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

multipart_data = MultipartEncoder(
    fields={
            # a file upload field
            'file': ('NoisedFile.wav', open('videoplayback.wav', 'rb'))
           }
    )
    
response = requests.post('172.26.33.199:5000', data=multipart_data,
                  headers={'Content-Type': multipart_data.content_type})

with open("a.wav", "wb") as f:
    f.write(response.content)
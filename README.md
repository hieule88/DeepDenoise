# ENVIROMENT

pip install asteroid
torch==1.8.0
librosa==0.8.0

# HOW TO RUN:
# Step 1:
Run serve.py

# Step 2:
In the client.py file change the Inputfile.wav to your Input dir:

multipart_data = MultipartEncoder(
    fields={
            # a file upload field
            # Input file
            'file': ('Inputfile.wav', open('Inputfile.wav', 'rb'))
           }
    )
response = requests.post('http://localhost:5000', data=multipart_data,
                  headers={'Content-Type': multipart_data.content_type})

# Write denoised wav to a.wav
with open("a.wav", "wb") as f:
    f.write(response.content)
    
# Step 3:
Run client.py

Then the denoised wav will save to a.wav 

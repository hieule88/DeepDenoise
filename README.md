# USING DCCRN
# ENVIROMENT

pip install asteroid
torch==1.8.0
librosa==0.8.0

# HOW TO RUN:
# Step 1:
Run serve.py

# Step 2:
In the client.py file change the Inputfile.wav to your Input dir:

# Write denoised wav to a.wav
with open("a.wav", "wb") as f:
    f.write(response.content)
    
# Step 3:
Run client.py

Then the denoised wav will save to a.wav 

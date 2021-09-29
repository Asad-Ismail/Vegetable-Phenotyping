from pepper_segment_orient_pred import *
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse




print(f"Intializing Model!!")
model_path="/media/asad/ADAS_CV/vegs_results/models/pepper/keypoints/model_final.pth"
pepp=detectroninference(model_path)
print(f"Intializing Model Done!!")

app = FastAPI(title='Deploying Pepper Phenotyping with FastAPI')


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

    print(f"Performing Prediction!!")
    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # 3. RUN Pepper and keypoint detection
    print(f"Running the prediction on Image!!")
    detected_cucumber,all_masks,all_patches,boxes,keypoints,*_=pepp.pred(image)
    # Add more results in dictionary as required
    results={}
    results["Detected"]=detected_cucumber.tolist()
    
    # Save it in a folder within the server
    #cv2.imwrite(f'images_uploaded/{filename}', detected_cucumber)
    
    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    
    # Open the saved image for reading in binary mode
    #file_image = open(f'images_uploaded/{filename}', mode="rb")

    # Return the results as json encodded
    json_compatible_item_data = jsonable_encoder(results)
    return JSONResponse(content=json_compatible_item_data)
    # Return the image as a stream specifying media type
    #return StreamingResponse(file_image, media_type="image/jpeg")


nest_asyncio.apply()
host =  "127.0.0.1"
uvicorn.run(app, host=host, port=8000)
import os
import io
import cv2
import requests
import numpy as np
import json


base_url = 'http://localhost:8000'
endpoint = '/predict'
url_with_endpoint_no_params = base_url + endpoint

# If we need to pass parameter
#full_url = url_with_endpoint_no_params + "?model=" + model
#full_url

def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """
    
    files = {'file': image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    if verbose:
        msg = "Everything went well!" if status_code == 200 else "There was an error when handling the request."
        print(msg)
    return response

def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after pepper detection.
    """
    #print(f"Response is {response.content}")
    out=json.loads(response.content.decode('utf-8'))
    #print(f"The shape of Response Image is {out['Detected']}")
    detected=np.array(out['Detected'],dtype=np.uint8)
    cv2.imshow("Output",detected)
    cv2.waitKey(0)
    #image_stream = io.BytesIO(response.content)
    #image_stream.seek(0)
    #file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    #image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #filename = "image_with_objects.jpeg"
    #cv2.imwrite(f'images_predicted/{filename}', image)


image_path="images/test.jpg"

with open(image_path, "rb") as image_file:
    prediction = response_from_server(url_with_endpoint_no_params, image_file)

display_image_from_response(prediction)
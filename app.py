import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import io
import threading
import base64
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input , decode_predictions
from keras.models import load_model
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Import your machine learning model module


app = Flask(__name__)

local = threading.local()

def load_img(path, target_size):
    img = Image.open(path)
    img = img.resize(target_size)
    return img
    
def img_to_array(img):
    return np.array(img)
    
@app.route('/')
def home():
    return render_template('main_page_heatmap.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'image' not in request.files:
        return render_template('main_page_heatmap.html', message='No image file uploaded')
    
    model = load_model('vgg16_model_for_heatmap.h5')
    # Read the image file from the request
    image_file = request.files['image']
    img_data = base64.b64encode(image_file.read()).decode()
    # Ensure the file is a valid image
    if image_file.filename == '':
        return render_template('main_page_heatmap.html', message='No image file selected')
    
    # Read the image data and convert it to a format suitable for processing
    imgs = load_img(image_file, target_size=(224,224))
    img_np = img_to_array(imgs)
    img_t = np.expand_dims(img_np , axis=0)
    img_t = img_t.astype('float32')
    img_t = preprocess_input(img_t)
    
    
    # Perform inference using your machine learning model
    pred = model.predict(img_t)
    
    values = decode_predictions(pred , top=3) [0]
    first_pred = values[0][1]
    second_pred = values[1][1]
    third_pred = values[2][1]
    
    index = np.argmax(pred[0])
    
    outputs = model.output[:,index] 
    last_layer = model.get_layer('block5_conv3')

    grads = K.gradients(outputs , last_layer.output) [0]
    pooled_grads = K.mean(grads , axis=(0,1,2))

    iterate = K.function(model.input , [pooled_grads , last_layer.output [0]])
    grd , feature_m = iterate(img_t)

    for i in range(512):
        feature_m[:,:,i] *= grd[i]

    heatmap = np.mean(feature_m , axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    
    # Convert the output image to a data URL for displaying in the browser
    output_image_data_url = image_to_data_url(img_np, heatmap)
    data = "Image overlayed with a heatmap illustrating the key regions that significantly influenced the model's predictions"
    # Pass the output image data URL to the template
    return render_template('main_page_heatmap.html', output_image=output_image_data_url , text=data, first=first_pred , second=second_pred , third=third_pred)
    
def image_to_data_url(image , heatmap):
    """Converts a PIL Image to a data URL."""
    # with io.BytesIO() as buffer:
        # Convert image to mode 'RGB' before saving as PNG
    # image_data = image.read()
    # nparr = np.frombuffer(image_data, np.uint8)
    # try:
    #   img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # except Exception as e:
    #   print("Error decoding image:", e)
    
    # heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # superimposed_img = heatmap * 0.5 + img
    # superimposed_img = np.clip(superimposed_img , 0 , 255)
    # cv2.imshow(superimposed_img)
    
    # # Convert the figure to a PNG image
    # buffer = io.BytesIO()
    # cv2.imwrite(buffer, image, format='png')
    # buffer.seek(0)
        
    # data_uri = base64.b64encode(buffer.getvalue()).decode()
    # return 'data:image/png;base64,' + data_uri

    # Resize the heatmap to match the dimensions of the image
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply color map to heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * 0.5 + image
    superimposed_img = np.clip(superimposed_img , 0 , 255)

    # Convert the superimposed image to a PNG format
    success, buffer = cv2.imencode('.png', superimposed_img)
    if not success:
        print("Error encoding image to PNG format")
        return None

    # Encode the PNG image data as base64
    data_uri = base64.b64encode(buffer).decode()
    
    return 'data:image/png;base64,' + data_uri

if __name__ == '__main__':
    app.run(debug=True)

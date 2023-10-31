# import the opencv library
import cv2
import numpy as np
import tensorflow as tf

# define a video capture object
vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model('keras_model.h5')

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    #redimensionar imagem
    img = cv2.resize((frame),224,224,3)
    #converta a imagem em uma aray numpy e aumente a dimensão
    test_image = np.array(img,dytipe=np.float32)
    test_image=np.expand_dims(test_image,axis=0)
    #normalize a imagem
    normalised_image = test_image/255.0

    #preveja o resultado
    prediction = model.predict(normalised_image)
    print("previsão: ", prediction)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()
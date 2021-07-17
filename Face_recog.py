from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import cv2 as cv
import numpy as np
import os

def extract_face(filename):
    img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)

    detector = MTCNN()
    results = detector.detect_faces(img)

    x1, y1, w, h = results[0]['box']
    x2, y2 = x1 + w, y1 + h 

    frame = img.copy()

    cv.rectangle(frame, (x1,y1), (x2,y2), (255,255,0))

    face = img[y1:y2, x1:x2]
    face = cv.resize(face, (224,224))

    face_array = np.asarray(face)
    return face_array

def get_embeddings(filenames):

    faces = [extract_face("people/"+ f) for f in filenames]
    
    samples = np.asarray(faces, 'float32')    
    samples = preprocess_input(samples, version=2)

    model = VGGFace(model="resnet50", include_top=False, input_shape=(224,224,3), pooling='avg')

    yhat = model.predict(samples)
    
    return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
 
def show_detection(files, known, candidate):
    know = extract_face("people/" + files[known])
    candi = extract_face("people/" + files[candidate])
    cv.imshow("known", know)
    cv.imshow("candidate", candi)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

files = os.listdir("people")

embeddings = get_embeddings(files)

show_detection(files, 0,6)
is_match(embeddings[0], embeddings[6])



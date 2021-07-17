from tensorflow import keras
import cv2 as cv
import numpy as np
import segmentation_models as sm

IMG_SIZE = 256

model = keras.models.load_model("neww.h5", compile=False)

model.compile( 'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score])

img = cv.imread("/home/blackpanther/Downloads/BluePart5.bmp")
img = cv.resize(img, (IMG_SIZE,IMG_SIZE))
img  = np.expand_dims(img, 0)

print(img.shape)


pred = model.predict(img)
# predThresh = (pred>0.1).astype(np.uint8)
print(np.unique(pred))

cv.imshow("Result", np.squeeze(pred))
cv.waitKey(0)


# if some numpy error, install numpy through anaconda prompt for this environment; pip install six also might be required
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2

# only uses 1 factor, check colab for 2 factor approach

model = load_model(r'C:\Users\Administrator\PycharmProjects\Personal_Face_Mask_Detection_V3_RT\model')
vs = VideoStream(src=0).start()
while True:
    faces = []
    frame = vs.read()
    frame = imutils.resize(frame, width=650)
    frame = cv2.flip(frame, 1)
    face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face, mode='tf')
    faces.append(face)
    faces = np.array(faces, dtype="float32")
    preds = model.predict(faces)
    cv2.putText(frame, str(preds) + ('without mask' if model.predict(faces)[0][0] < 0.90 else 'with mask'), (20, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
    cv2.rectangle(frame, (162, 50), (475, 450), (0, 0, 0), 2)
    cv2.imshow("Mask Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()

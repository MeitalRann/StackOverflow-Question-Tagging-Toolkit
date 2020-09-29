import os.path
from thensorflow.keras.models import load_model

if os.path.isfile('model.h5') is false:
    model.save('model.h5')

old_model = load_model('model.h5')
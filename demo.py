from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS
from neuralforecast.utils import AirPassengersDF
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


nf = NeuralForecast(
    models = [NBEATS(input_size=24, h=12, max_steps=100)],
    freq = 'ME'
)

nf.fit(df=AirPassengersDF)
nf.predict()
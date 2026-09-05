import librosa
import numpy as np
from dotenv import load_dotenv
import os
import json
from easydict import EasyDict as edict
with open('src/config.json') as f:
    config = edict(json.load(f))
seed = config.random_seed
import random
random.seed(seed)

import warnings
from typing import Tuple, Union, List
import logging
from enum import Enum
from PIL import Image
import matplotlib.pyplot as plt
import soundfile as sf


warnings.filterwarnings("ignore")
load_dotenv()

SAMPLING_RATE = int(os.getenv("SAMPLING_RATE"))
PROCESSED_DIR = os.getenv("PROCESSED_DATA_DIR")
RAW_WAVEFORM_PATH = os.getenv("RAW_WAVEFORM_PATH")
CROPPED_DIR = os.getenv("CROPPED_DATA_DIR")

class SpectrogramType(Enum):
    MEL = "mel"
    STFT = "stft"
    CQT = "cqt"
    FFT = "fft"

class NormalizeType(Enum):
    RMS = "rms"
    STANDARD = "std"

class SpectogramProcessor:
    def __init__(self,
                 samplingRate: int = SAMPLING_RATE,
                 mono: bool = False,
                 normalize: bool = True,
                augment: bool = False,
                crop: bool = True,
                base: bool = False,
                duration: int = 30,
                cropDuration: int = 15,
                spectrogramType: SpectrogramType = SpectrogramType.MEL,
                preprocessing: List[str] = ["normalize:std",],
                **spectogramKwargs):
        
        self.samplingRate = samplingRate
        self.mono=mono
        self.normalize = normalize
        self.augment = augment
        self.base = base
        self.crop= crop
        self.duration = duration
        self.cropDuration = cropDuration
        self.spectrogramType = spectrogramType
        self.spectogramKwargs = spectogramKwargs
        self.proprocessing = preprocessing



        # any other types?
        self.spectrogramFunctions = {
            SpectrogramType.MEL: self.toMel,
            SpectrogramType.STFT : self.toStft,
            SpectrogramType.CQT: self.toCqt,
            SpectrogramType.FFT: self.toFft
        }

        self.normalizeFunctions = {
            NormalizeType.RMS.value: self.rmsNormalize,
            NormalizeType.STANDARD.value : self.stdNormalize
        }

        # logging setup
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # make dir if doesnt exist
        if not os.path.exists(PROCESSED_DIR):
            os.makedirs(PROCESSED_DIR)
            self.logger.info("Created ditectory for processed data")
        else:
            self.logger.info("Directory for processed data already exists")
        

    def loadAudio(self,file) -> Tuple[int, np.ndarray]:
        y,sr =librosa.load(file) #mono=self.mono,sr=self.samplingRate)
        return y,sr
    
    def toMono(self, y):
        mono = librosa.to_mono(y)
        return mono
    

    def trimOrPad(self,y):
        finalLen = int(self.samplingRate * self.duration)

        if len(y)> finalLen:
            return y[:finalLen]
        elif len(y)<finalLen:
            return np.pad(y,(0,finalLen-len(y)))
        return y
    
    def cropAudio(self,y, sr):
        samplesToKeep = int(self.cropDuration * sr)
        yCropped = y[:samplesToKeep]
        return yCropped

    def toMel(self, y):
        return librosa.feature.melspectrogram(y=y, **self.spectogramKwargs) # sr=self.samplingRate,

    def toStft(self, y):
        return np.abs(librosa.stft(y, **self.spectogramKwargs))

    def toCqt(self, y):
        return np.abs(librosa.cqt(y, sr=self.samplingRate))

    def toFft(self, y):
        return np.abs(np.fft.rfft(y))[:, None]  # FFT is 1D; we reshape to 2D to align with other outputs and aim for consistency
    
    def generateSpectrogram(self, y):
        spectrogramFunction = self.spectrogramFunctions.get(self.spectrogramType) # from a dict get the function according to the type

        if not spectrogramFunction:
            raise ValueError(f"This spectrogram type doesn't exist: {self.spectrogramType}, choose one of these types: {', '.join(t.value for t in SpectrogramType)}")
        
        spect = spectrogramFunction(y)
        return spect
    
    def rmsNormalize(self,y, targetRms= 0.1):
        rms = np.sqrt(np.mean(y ** 2))
        scalingFactor = targetRms / (rms + 1e-6)
        normalizedWaveforms = y * scalingFactor
        return normalizedWaveforms
    
    def stdNormalize(self,y):
        mean = np.mean(y)
        stdDev = np.std(y)
        normalizedWaveforms = (y - mean) / stdDev
        return normalizedWaveforms
    
    def pitchShifting(self, y):
        nSteps =5
        yShifted = librosa.effects.pitch_shift(y, sr= self.samplingRate, n_steps=nSteps)
        return yShifted
    
    def addNoise(self,y, noiseLevel = 0.1):
        noise = np.random.normal(0, noiseLevel, len(y))
        yNoisy = y + noise
        return yNoisy


    def applyProcessing(self,y):
        for processingStep in self.proprocessing:

            if "normalize" in processingStep:
                _,method = processingStep.strip().split(":")
                normalizeFunction = self.normalizeFunctions.get(method)
                y = normalizeFunction(y)
            if processingStep == "addNoise":
                y = self.addNoise(y)
            if processingStep == "pitchShifting":
                y = self.pitchShifting(y)
        return y

    def plotSpectrogram(self, generatedSpect, sr):
        fig, ax = plt.subplots()
        S_dB = librosa.power_to_db(generatedSpect, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr,fmax=8000, ax=ax)
        ax.set_axis_off()
        return fig

    def processAndSave(self):
        genres = os.listdir(RAW_WAVEFORM_PATH)

        for genre in genres:
            self.logger.info(f"Loading audio for {genre}")
            genreDir = os.path.join(RAW_WAVEFORM_PATH,genre) # raw directory path
            genreDirProcessed = os.path.join(PROCESSED_DIR,genre) # processed directory path
            genreDirCropped = os.path.join(CROPPED_DIR, genre)
            os.makedirs(genreDirProcessed,exist_ok=True)
            os.makedirs(genreDirCropped,exist_ok=True)
            genreFiles  = os.listdir(genreDir)
            
            if self.base:
                for file in genreFiles:
                    audioPath = os.path.join(genreDir, file)
                    try:
                        audioVals, sr = self.loadAudio(audioPath)
                    except Exception as e:
                        print(f"Could not load {audioPath}: {e}")
                        continue  # Skip this file and move to the next

                    y = self.trimOrPad(audioVals)
                    y = self.applyProcessing(y)  # Denoising and other processing
                    generatedSpect = self.generateSpectrogram(y)

                    fig = self.plotSpectrogram(generatedSpect, sr)
                    fig.savefig(os.path.join(genreDirProcessed, f"{file[:-4]}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig) 

            # after transformign original, check if we should augument
            if self.augment: 
                newImgsFiles = random.sample(genreFiles, 20)
                for fileName in newImgsFiles:
                    audioPath = os.path.join(genreDir, fileName)
                    try:
                        audioVals, sr = self.loadAudio(audioPath)
                    except Exception as e:
                        print(f"Could not load {audioPath}: {e}")
                        continue
                    y = self.trimOrPad(audioVals)
                    y = self.applyProcessing(y)
                    y = self.addNoise(y)
                    y = self.pitchShifting(y)
                    generatedSpect = self.generateSpectrogram(y)

                    fig = self.plotSpectrogram(generatedSpect, sr)
                    fig.savefig(os.path.join(genreDirProcessed, f"{fileName[:-4]}aug.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig) 

            if self.crop:
                for file in genreFiles[:10]:
                    audioPath = os.path.join(genreDir, file)
                    try:
                        audioVals, sr = self.loadAudio(audioPath)
                    except Exception as e:
                        print(f"Could not load {audioPath}: {e}")
                        continue

                    y = self.cropAudio(audioVals, sr)
                    y = self.trimOrPad(y)
                    y = self.applyProcessing(y)
                    generatedCropSpect = self.generateSpectrogram(y)

                    fig = self.plotSpectrogram(generatedCropSpect, sr)
                    fig.savefig(os.path.join(genreDirCropped, f"{file[:-4]}.png"), bbox_inches='tight', pad_inches=0)
                    plt.close(fig) 

                    

                    





if __name__== "__main__":
    obj = SpectogramProcessor(
    sr=22050,
    n_fft=2048, 
    hop_length=512,
    win_length=2048,
    window="hann",
    power=2.0,
    center=True,
    pad_mode="reflect"
)

    obj.processAndSave()


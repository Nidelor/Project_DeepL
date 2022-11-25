
import soundfile, torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.transforms import Resample

audiofile = "sound_library\sea_sound.wav"
print(torchaudio.info(audiofile))
X,sr = torchaudio.load(audiofile)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=True)

#plot_waveform(X, sr)

# Check that the sample rate is the same as the trained SoundNet
if sr != 22050:
    transform = Resample(sr,22050)
    X = transform(X)

# X is now a torch FloatTensor of shape (Number of Channels, Number of samples)
# A stereophonic recording will have 2 channels ; SoundNet only accepts monophonic so we average the two channels if necessary
if X.shape[0]>1:
    X = torch.mean(X,axis=0)

from pytorch_model import SoundNet8_pytorch
from utils import vector_to_scenes,vector_to_obj

## define the soundnet model
model = SoundNet8_pytorch()

## Load the weights of the pretrained model
model.load_state_dict(torch.load('sound8.pth'))

# Reshape the data to the format expected by the model (Batch, 1, time, 1)
X = X.view(1,1,-1,1)
    
# Compute the predictions of Objects and Scenes
object_pred, scene_pred = model(X)

## Shape of object_pred is (1,1000,TIME,1) as there are 1000 possible objects
## Find the correspond object labels, and print it for each time point
print(vector_to_obj(object_pred.detach().numpy()))

# Shape of scene_pred is (1,401,TIME,1) as there are 401 possible scenes
# Find the correspond scene label, and print it  for each time point
print(vector_to_scenes(scene_pred.detach().numpy()))

# Extract internal features
features = model.extract_feat(X)

# Features is a List of Tensors, each element of this list corresponds to a layer of SoundNet. From 0 to 6 -> conv1 to conv7, 7 -> conv of object prediction and 8 -> conv of scene prediction. See the extract_feat method in the model code.

## Example : Feature maps of Layer Conv5 is of shape : (Batch, Units, Time, 1)
print((features[4].shape))
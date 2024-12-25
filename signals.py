import wave 
import numpy as np

'''Loads the .wav file with skipping based on timesteps'''
def load_audio_signal(length, output_filename="signal.wav"):
    with wave.open(output_filename, 'rb') as wf:
        # Extract audio parameters
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        # Read frames and convert to numpy array
        audio_data = wf.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo, reshape array to separate channels
        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels)
        
        # Normalize audio signal and account for incomplete reads
        # audio_array is the non normalized signal
        max_value = np.max(audio_array)
        normalized_audio_array = audio_array / max_value
        indices = np.linspace(0, audio_array.size - 1, num=length, dtype=int) # Want integers (non-decimal values)
        skipped_audio_array = normalized_audio_array[indices]

    return skipped_audio_array


'''Generate constant signal'''
def load_constant_signal(length, value, output_filename="signal.wav"):
    signal = np.zeros(length) + value
    return signal


''''Generates a random binary signal'''
def load_digital_signal(length, amplitude, pulse):
    signal = np.zeros(length)
    for index in range(0, length, pulse):
        signal[index] = amplitude*np.round(np.random.rand())
        
    return signal
    
import pyaudio # Need pyaudio installed for recording
import wave 


''''Function for recording an audio signal for use'''
def record_audio_signal(duration, timesteps, output_filename="signal.wav"):
    # Parameters for recording
    sample_rate = float(timesteps)/duration  # Sample rate in Hz
    channels = 1  # Not stereo
    chunk = 1024  # Size of each buffer chunk

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording...")

    # Record data in chunks
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Recording complete.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save recorded data to a .wav file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"Audio saved as '{output_filename}'.")
    
    return output_filename


'''Record generic audio signal (voice or song)'''
def main():
    record_audio_signal(duration=100, timesteps=10000)
    

if __name__ == "__main__":
    main()
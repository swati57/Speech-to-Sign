
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
# import wavio as wv
# import sounddevice as sd
# from scipy.io.wavfile import write
# import scipy.io.wavfile as wavfile
# from huggingface_hub import from_pretrained_keras

def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

model = tf.keras.models.load_model('model30.h5', custom_objects={'CTCLoss':CTCLoss})



characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384

SAMPLE_RATE = 22050


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def load_16k_audio_wav(filename):
    # Read file content
    file_content = tf.io.read_file(filename)

    # Decode audio wave
    audio_wav, sample_rate = tf.audio.decode_wav(file_content, desired_channels=1)
    audio_wav = tf.squeeze(audio_wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)

    # Resample to 16k
    audio_wav = tfio.audio.resample(
        audio_wav, rate_in=sample_rate, rate_out=SAMPLE_RATE
    )

    return audio_wav


def mic_to_tensor(recorded_audio_file):
    sample_rate, audio = recorded_audio_file

    audio_wav = tf.constant(audio, dtype=tf.float32)
    if tf.rank(audio_wav) > 1:
        audio_wav = tf.reduce_mean(audio_wav, axis=1)
    audio_wav = tfio.audio.resample(
        audio_wav, rate_in=sample_rate, rate_out=SAMPLE_RATE
    )

    audio_wav = tf.divide(audio_wav, tf.reduce_max(tf.abs(audio_wav)))

    return audio_wav


def tensor_to_predictions(audio_tensor):
    # 3. Change type to float
    audio_tensor = tf.cast(audio_tensor, tf.float32)

    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio_tensor,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
    )

    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)

    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    spectrogram = tf.expand_dims(spectrogram, axis=0)

    batch_predictions = model.predict(spectrogram)
    batch_predictions = decode_batch_predictions(batch_predictions)
    return batch_predictions


def clear_inputs_and_outputs():
    return [None, None, None]


def predict( uploaded_audio_file):
    # 1. Read wav file
    
    audio_tensor = load_16k_audio_wav(uploaded_audio_file)

    prediction = tensor_to_predictions(audio_tensor)[0]
    
    return prediction


# print(predict('test.wav'))

# def record():

#         fs = 44100  # Sample rate
#         seconds = 10  # Duration of recording
#         myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
#         sd.wait()  # Wait until recording is finished
#         wv.write("recording1.wav", myrecording, fs, sampwidth=2)

#         samplerate, data = wavfile.read('recording1.wav')
#         write('output1.wav', fs, data.astype(np.int16))  # Save as WAV file
        


# print("Recording")

# record()
# print(predict('recording1.wav'))
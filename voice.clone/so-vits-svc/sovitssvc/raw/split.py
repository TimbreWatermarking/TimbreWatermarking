from pydub import AudioSegment

def split_audio(audio_path, interval=10, output_format="wav"):
    # Load audio file
    audio = AudioSegment.from_file(audio_path)

    # Length of audio in milliseconds
    length_ms = len(audio)

    # Convert interval to milliseconds
    interval_ms = interval * 1000

    # Calculate number of chunks
    num_chunks = length_ms // interval_ms

    # Split audio
    for i in range(1, num_chunks):
        start_time = i * interval_ms
        end_time = (i + 1) * interval_ms
        chunk = audio[start_time:end_time]
        chunk.export(f"chunk_{i}.{output_format}", format=output_format)
        
    # Handle leftover segment
    if length_ms % interval_ms != 0:
        chunk = audio[num_chunks*interval_ms:]
        chunk.export(f"chunk_{num_chunks}.{output_format}", format=output_format)

# Example usage
# split_audio("1_Ta Bu Dong - AJ Zhang Jie_(Vocals).wav")
split_audio("1_3445150551_(Vocals).wav")


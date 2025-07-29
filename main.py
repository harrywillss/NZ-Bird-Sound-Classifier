import data_downloader
import process_audio

def main():
    # Download the audio data
    data_downloader.main()
    process_audio.main()

if __name__ == "__main__":
    main()
    print("Audio processing pipeline started.")
    print("All tasks completed successfully.")
    print("You can now check the processed audio files in the output directory.")
    print("Thank you for using the audio processing pipeline!")
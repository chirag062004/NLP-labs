# Importing the necessary libraries
import speech_recognition as sr  # For speech-to-text functionality
import keyboard  # For detecting keypresses

def recognize_speech_from_mic(recognizer, microphone):
    """Transcribe speech recorded from `microphone`."""

    # Ensure `recognizer` is an instance of sr.Recognizer
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    # Ensure `microphone` is an instance of sr.Microphone
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # Use the microphone as the source of audio input
    with microphone as source:
        print("Adjusting for ambient noise...")
        # Adjust for ambient noise for 1 second to improve recognition accuracy
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Ready to listen... (Press 'q' to quit)")
        # Listen from the microphone with a phrase time limit of 5 seconds
        audio = recognizer.listen(source, phrase_time_limit=5)

    # Dictionary to store the response
    response = {
        "success": True,          # Indicates if API call was successful
        "error": None,            # Stores any error messages
        "transcription": None     # Will hold the transcribed text
    }

    try:
        # Try to recognize the speech using Google Web Speech API
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # If the API is unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # If speech could not be understood
        response["error"] = "Unable to recognize speech"

    return response

def main():
    # Create instances of Recognizer and Microphone
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    # Display instructions to the user
    print("Speech to Text Converter")
    print("-----------------------")
    print("Speak clearly into your microphone after the 'Ready to listen...' prompt.")
    print("Press 'q' to exit or Ctrl+C to exit.\n")

    try:
        while True:
            # Check if the 'q' key has been pressed to exit
            if keyboard.is_pressed('q'):
                print("\n'q' pressed. Exiting program.")
                break

            # Capture and process the speech
            result = recognize_speech_from_mic(recognizer, microphone)

            # If speech was recognized, print it
            if result["transcription"]:
                print(f"You said: {result['transcription']}")
            # If there was an error, print the error message
            elif result["error"]:
                print(f"ERROR: {result['error']}")
            # If nothing was heard
            else:
                print("No speech detected. Please try again.")

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nExiting program.")

if __name__ == "__main__":
    # Ensure the keyboard package is available (redundant here since already imported)
    try:
        import keyboard
    except ImportError:
        # If the keyboard package is missing, install it using pip
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
        import keyboard

    # Call the main function to run the program
    main()

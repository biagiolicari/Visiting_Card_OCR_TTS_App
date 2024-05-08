# Visiting Card OCR and Text-to-Speech App

This application utilizes computer vision techniques to detect visiting cards from a live webcam feed, extract text from them using Azure Cognitive Services OCR (Optical Character Recognition), and then convert the extracted text to speech. 

## Features

- **Real-time Visiting Card Detection**: The application detects visiting cards in real-time from a live webcam feed using the YOLO (You Only Look Once) object detection model.
  
- **Text Extraction**: After detecting a visiting card, the application extracts the text from it using Azure Cognitive Services OCR API.

- **Text-to-Speech Conversion**: Once the text is extracted, the application converts it to speech using Azure Cognitive Services Text-to-Speech API.

- **Quality-based Frame Selection**: The application evaluates the quality of each frame to select the best frame for OCR processing, ensuring accurate text extraction.

## Requirements

- Python 3.6 or higher
- OpenCV
- aiohttp
- azure-cognitiveservices-speech
- ultralytics

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/visiting-card-ocr.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up Azure Cognitive Services:

    - Sign up for an Azure account if you don't have one already.
    - Obtain the subscription key and region for Azure Cognitive Services Speech and Vision APIs.
    - Set these values in the `.env` file:

        ```
        COG_SERVICE_ENDPOINT=<your_cognitive_service_endpoint>
        COG_SERVICE_KEY=<your_cognitive_service_key>
        SPEECH_REGION=<speech_region>
        ```

## Usage

1. Run the application:

    ```bash
    python main.py
    ```

2. Point your webcam towards visiting cards to detect and extract text from them.
3. Press 'q' to quit the application.

## Notes

- Adjust the confidence threshold (`threshold` variable in `main.py`) for object detection as needed.
- Ensure that the webcam is properly connected and configured before running the application.

## Credits

- YOLO (You Only Look Once) object detection model is implemented using the ultralytics library.
- Azure Cognitive Services is used for OCR (Optical Character Recognition) and Text-to-Speech conversion.
- Roboflow 
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Feel free to customize the README according to your specific project details and preferences.

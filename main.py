import asyncio
import os

import aiohttp
import azure.cognitiveservices.speech as speechsdk
import cv2
from dotenv import load_dotenv
from ultralytics import YOLO


async def text_to_speech(text, subscription_key, region):
    speech_config = speechsdk.SpeechConfig(subscription=subscription_key,
                                           region=region)
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # The neural multilingual voice can speak different languages based on the input text.
    speech_config.speech_synthesis_voice_name = 'it-IT-ElsaNeural'

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


def process_ocr_response(response_data):
    text_lines = []
    for region in response_data['regions']:
        for line in region['lines']:
            text_lines.append(' '.join([word['text'] for word in line['words']]))
    extracted_text = '\n'.join(text_lines)
    return extracted_text


async def extract_text_from_image(cropped_img, endpoint, subscription_key):
    print('Reading text')
    detect_orientation = True

    # Request parameters
    params = {
        'overload': 'stream',
        'detectOrientation': str(detect_orientation).lower()
    }

    # Request headers
    headers = {
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key,
    }

    # Azure Cognitive Services endpoint for OCR
    endpoint_url = f'{endpoint}/vision/v3.1/ocr'

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint_url, params=params, data=image_bytes, headers=headers) as response:
            # Wait for the response
            response_data = await response.json()
            print(response_data)

            # Process the response
            # Extract the text from the response data
            extracted_text = process_ocr_response(response_data)

            return extracted_text


def evaluate_frame_quality(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate Laplacian variance as a measure of sharpness
    sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()

    # Calculate image contrast
    min_val, max_val, _, _ = cv2.minMaxLoc(gray_frame)
    contrast = max_val - min_val

    # Combine sharpness and contrast to compute overall quality score
    quality_score = sharpness * contrast

    return quality_score


def get_best_frame(image_batch):
    best_frame_index = 0
    best_quality_score = float('-inf')

    # Iterate over each frame in the batch
    for i, frame in enumerate(image_batch):
        quality_score = evaluate_frame_quality(frame)

        # Check if the current frame has higher quality than the previous best frame
        if quality_score > best_quality_score:
            best_quality_score, best_frame_index = quality_score, i

    return image_batch[best_frame_index]


async def process_frame(frame, model, threshold, endpoint, subscription_key, region, image_batch, batch_size):
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            # Get coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Check confidence threshold
            if confidence > threshold:
                # Draw rectangle around detected object
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Extract the region of interest (ROI)
                cropped_img = frame[y1:y2, x1:x2]

                # Do not consider if too blurry
                if cv2.Laplacian(cropped_img, cv2.CV_64F).var() >= 230:
                    # Display ROI
                    cv2.imshow('Visiting Card', cropped_img)

                    # Add frame to image batch
                    # Convert the cropped image (NumPy array) into bytes
                    cropped_img = cv2.imencode('.jpg', cropped_img)[1].tobytes()
                    image_batch.append(cropped_img)

    # Check batch size and process if reached
    if len(image_batch) >= batch_size:
        best_frame = get_best_frame(image_batch)

        text = await extract_text_from_image(best_frame, endpoint=endpoint,
                                             subscription_key=subscription_key)
        await text_to_speech(text, subscription_key, region)

        # Clear the batch
        image_batch.clear()


async def main(model, threshold, endpoint, subscription_key, region):
    cap = cv2.VideoCapture(0)  # Use the default camera
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    batch_size = 5
    image_batch = []

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Error: Unable to capture frame.")
                continue

            await process_frame(img, model, threshold, endpoint, subscription_key, region, image_batch, batch_size)

            cv2.imshow('Webcam', img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    load_dotenv()
    cog_endpoint = os.getenv('COG_SERVICE_ENDPOINT')
    cog_key = os.getenv('COG_SERVICE_KEY')
    region = os.environ.get('SPEECH_REGION')

    # Load YOLO model trained
    model = YOLO("yolov8_visiting_card.pt")

    asyncio.run(
        main(model, threshold=0.97, endpoint=cog_endpoint, subscription_key=cog_key, region=region))

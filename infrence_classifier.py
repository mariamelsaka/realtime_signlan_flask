import pickle
import mediapipe as mp
import numpy as np
import cv2
from bidi.algorithm import get_display
import arabic_reshaper
from PIL import ImageFont, ImageDraw, Image

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {4: 'alef',
               5: 'bah', 6: 'tah', 3: 'thah', 0: 'incorrect', 1: 'incorrect', 2: 'incorrect'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character == 'alef':
            text = "الف"
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 50)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text_size = draw.textsize(bidi_text, font=font)
            text_x = int((W - text_size[0]) / 2)
            text_y = int((H - text_size[1]) / 2)
            draw.text((text_x, text_y), bidi_text, font=font, fill=(255, 255, 255))
            frame = np.array(img_pil)
        elif predicted_character == 'bah':
            text = "باء"
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 50)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text_size = draw.textsize(bidi_text, font=font)
            text_x = int((W - text_size[0]) / 2)
            text_y = int((H - text_size[1]) / 2)
            draw.text((text_x, text_y), bidi_text, font=font, fill=(255, 255, 255))
            frame = np.array(img_pil)
        elif predicted_character == 'tah':
            text = "تاء"
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 50)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text_size = draw.textsize(bidi_text, font=font)
            text_x = int((W - text_size[0]) / 2)
            text_y = int((H - text_size[1]) / 2)
            draw.text((text_x, text_y), bidi_text, font=font, fill=(255, 255, 255))
            frame = np.array(img_pil)
        elif predicted_character == 'thah':
            text = "ثاء"
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath, 50)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text_size = draw.textsize(bidi_text, font=font)
            text_x = int((W - text_size[0]) / 2)
            text_y = int((H - text_size[1]) / 2)
            draw.text((text_x, text_y), bidi_text, font=font, fill=(255, 255, 255))
            frame = np.array(img_pil)
        elif predicted_character == 'incorrect':
            text = "خاطئ"
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            fontpath = "arial.ttf"
            font = ImageFont.truetype(fontpath,50)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            text_size = draw.textsize(bidi_text, font=font)
            text_x = int((W - text_size[0]) / 2)
            text_y = int((H - text_size[1]) / 2)
            draw.text((text_x, text_y), bidi_text, font=font, fill=(255, 255, 255))
            frame = np.array(img_pil)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
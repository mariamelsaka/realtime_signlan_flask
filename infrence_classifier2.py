import pickle
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display
import arabic_reshaper
import cv2
import mediapipe as mp
import numpy as np


model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']


# cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


labels_dict = {0: 'min',1: 'thanks',2: 'please',4:'good',5:'bad',
               6:'helove', 7:'dad',8:'mom', 9:'grand' ,10:'sad'
               ,11:'iloveu'}


class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        # do some processing on the frame here
        # while True:

        data_aux = []
        x_ = []
        y_ = []

        # ret, frame = cap.read()

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

            if predicted_character == 'min':
                text = "دقائق"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'thanks':
                text = "شكرا"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'please':
                text = "لو سمحت"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'good':
                text = "جيد"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'bad':
                text = "سيء"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'helove':
                text = "يحب"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'dad':
                text = "اب"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'mom':
                text = "امي"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'grand':
                text = "جد"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'sad':
                text = "محبط"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            elif predicted_character == 'iloveu':
                text = "احبك"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)
            else:
                text = "خاطئ"
                reshaped_text = arabic_reshaper.reshape(text)
                bidi_text = get_display(reshaped_text)
                fontpath = "arial.ttf"
                font = ImageFont.truetype(fontpath, 32)
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                text_size = draw.textsize(bidi_text, font=font)
                text_x = int((W - text_size[0]) / 2)
                text_y = int((H - text_size[1]) / 2)
                draw.text((text_x, text_y), bidi_text, font=font, fill=(0, 0, 0))
                frame = np.array(img_pil)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)



        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()


    def close(self):
        self.video.release()
        cv2.destroyAllWindows()

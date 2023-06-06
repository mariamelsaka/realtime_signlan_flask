from flask import Flask, render_template, Response
from infrence_classifier2 import Video


app = Flask(__name__,static_url_path='')

@app.route('/')
def index():
   return render_template("index.html")

def gen(infrence_classifier2):
	while True:
		frame = infrence_classifier2.get_frame()
		yield(b'--frame\r\n'
		b'Content-Type: image/jpeg\r\n\r\n' + frame +
		b'\r\n'
		)

@app.route('/video')

def video():
	return Response(gen(Video()),
	mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug = True)
#
#
#
#
#
#
# from flask import Flask, render_template, Response
# from infrence_classifier2 import Video
#
# app = Flask(__name__, static_url_path='')
#
# @app.route('/')
# def index():
#     return render_template("index.html")
#
# def gen(inference_classifier):
#     while True:
#         frame = inference_classifier.get_frame()
#         if frame is not None:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame +
#                    b'\r\n')
#         else:
#             break
#             print("wrong")
#
# @app.route('/video')
# def video():
#     return Response(gen(Video()), mimetype='multipart/x-mixed-replace; boundary=frame')
#
# if __name__ == '__main__':
#     app.run(debug=True)
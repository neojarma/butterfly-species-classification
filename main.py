from datetime import timedelta
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import cv2
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
# Replace the MySQL dialect with pymysql
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:secretrootpassword@localhost/butterfly'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:secretrootpassword@localhost/butterfly'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuration for JWT
app.config['JWT_SECRET_KEY'] = 'your-secret-keyetrbe4rtyrtyb678678'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=3)
jwt = JWTManager(app)

ALLOW_EXTENSION = {'jpg', 'jpeg'}

# Load the Machine Learning Model
model = tf.keras.models.load_model('trained_80_20.h5')
class_names = ['BROOKES BIRDWING', 'ELBOWED PIERROT', 'GREAT EGGFLY', 'GREAT JAY',
               'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'ULYSES']


class User(db.Model):
    __tablename__ = 'users'
    username = db.Column(db.String(80), primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    image_path = db.Column(db.String(255), nullable=False,
                           default='default_profil.jpg')


def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSION


def read_image(img):
    # Read the image
    image = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize the image
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_resized = np.expand_dims(image_resized, axis=0)
    return image_resized


@app.route('/')
def index():
    return "OK"


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    full_name = data.get('full_name')
    email = data.get('email')
    password = data.get('password')
    username = data.get('username')

    hashedPass = generate_password_hash(password)
    try:
        new_user = User(username=username, full_name=full_name,
                        email=email, password=hashedPass)
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'success register'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Username already exist'}), 409


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    password = data.get('password')
    username = data.get('username')

    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({'message': 'User not found'}), 404

    ok = check_password_hash(user.password, password)
    if not ok:
        return jsonify({'message': 'Wrong username or password'}), 401

    # Create a JWT token
    access_token = create_access_token(identity=username)
    user_dict = {
        'username': user.username,
        'full_name': user.full_name,
        'email': user.email,
        'image_path': user.image_path
    }

    return jsonify({'message': 'success login', 'data': user_dict, 'access_token': access_token})


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']

    if image and allow_file(image.filename):
        image = image.read()
        img = read_image(image)
        prediction = model.predict(img)
        print(prediction)
        output_class = class_names[np.argmax(prediction)]
        percentage = 100 * np.max(prediction)

        resp = jsonify({'class': output_class, 'percent': percentage})
        return resp


@app.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    # Access the identity of the current user with get_jwt_identity
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

from datetime import timedelta
import io
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import cv2
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import InvalidRequestError
import os
from PIL import Image
from sqlalchemy import ForeignKey, extract, or_

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:secretrootpassword@localhost/butterfly'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configuration for JWT
app.config['JWT_SECRET_KEY'] = 'your-secret-keyetrbe4rtyrtyb678678'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=3)
jwt = JWTManager(app)

host = 'https://growing-advanced-sculpin.ngrok-free.app'

ALLOW_EXTENSION = {'jpg', 'jpeg'}

# Load the Machine Learning Model
model = tf.keras.models.load_model('trained_80_20.h5')
class_names = ['brookes-birdwing', 'elbowed-pierrot', 'great-eggfly', 'great-jay',
               'orange-tip', 'orchard-swallow', 'painted-lady', 'paper-kite', 'peacock', 'ulyses']


class User(db.Model):
    __tablename__ = 'users'
    username = db.Column(db.String(255), primary_key=True)
    full_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False,
                           default='default_profil.jpg')


class Observation(db.Model):
    __tablename__ = 'observations'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    uploaded_by = db.Column(db.String(255))
    species = db.Column(db.String(255))
    total = db.Column(db.Integer)
    date = db.Column(db.Date)
    lat = db.Column(db.Float)
    lon = db.Column(db.Float)
    img_path = db.Column(db.String(255))


# get filter
@app.route('/species', methods=['GET'])
def allSpecies():
    return jsonify({'species': class_names})


@app.route('/observations', methods=['GET'])
def geographics():
    # Get the filter values from the query parameters
    species_filter = request.args.get('species')
    year_filter = request.args.get('year')
    month_filter = request.args.get('month')

    # Start building the base query
    query = Observation.query
    # Join with the User table to get full_name based on username
    query = query.join(User, User.username == Observation.uploaded_by)

    # Apply species filter if provided
    if species_filter:
        species_values = species_filter.split(',')
        query = query.filter(Observation.species.in_(species_values))

    # Apply year filter if provided
    if year_filter:
        query = query.filter(
            extract('year', Observation.date) == int(year_filter))

    # Apply month filter if provided
    if month_filter:
        query = query.filter(
            extract('month', Observation.date) == int(month_filter))
    query = query.with_entities(
        Observation.id,
        Observation.uploaded_by,
        User.full_name.label('full_name'),  # Alias for clarity
        Observation.species,
        Observation.total,
        Observation.date,
        Observation.lat,
        Observation.lon,
        Observation.img_path
    )
    # Execute the query and fetch the results
    observations = query.all()

    # Convert the results to a list of dictionaries
    observations_list = []
    for observation in observations:
        observation_dict = {
            'id': observation.id,
            'uploaded_by': observation.uploaded_by,
            'full_name': observation.full_name,
            'species': observation.species,
            'total': observation.total,
            'date': observation.date.strftime('%Y-%m-%d'),
            'lat': observation.lat,
            'lon': observation.lon,
            'observed_image': host+observation.img_path,
            'images_reference': genImagesPrev(observation.species),
        }
        observations_list.append(observation_dict)

    # Return the results as JSON
    return jsonify({'observations': observations_list})


def genImagesPrev(species):
    preview = []
    for i in range(1, 6):
        path = host + '/images'+'/butterflies/' + \
            str(species) + '/'+str(i)+'.jpg'
        preview.append(path)
    return preview


def allow_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOW_EXTENSION


def read_image(img):
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
        output_class = class_names[np.argmax(prediction)]
        percentage = 100 * np.max(prediction)

        preview = []
        for i in range(1, 6):
            path = host + '/images'+'/butterflies/' + \
                output_class+'/'+str(i)+'.jpg'
            preview.append(path)

        resp = jsonify(
            {'class': output_class, 'percent': percentage, 'images': preview})
        return resp


def saveImage(img_bytes, current_user, fileName):
    img = Image.open(io.BytesIO(img_bytes))

    file_path = os.path.join('images', 'observations', current_user)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = os.path.join(file_path, fileName)
    img.save(file_path)
    return '/'+file_path.replace('\\', '/')


@app.route("/observations", methods=["POST"])
@jwt_required()
def protected():
    current_user = get_jwt_identity()

    # Ensure that required fields are present in the request
    if 'species' not in request.form or 'total' not in request.form or 'date' not in request.form \
            or 'lat' not in request.form or 'lon' not in request.form or 'image' not in request.files:
        return jsonify({'error': 'Missing required data in the request'}), 400

    data = request.form
    uploaded_by = current_user
    species = data['species']
    total = int(data['total'])
    date = data['date']
    lat = float(data['lat'])
    lon = float(data['lon'])
    obsrv_image = request.files['image']

    fileName = obsrv_image.filename
    fullPath = saveImage(obsrv_image.read(), current_user, fileName)

    try:
        new_obs = Observation(uploaded_by=uploaded_by, species=species,
                              total=total, date=date, lat=lat, lon=lon, img_path=fullPath)
        db.session.add(new_obs)
        db.session.commit()
        return jsonify({'message': 'success'})
    except InvalidRequestError as e:
        error_message = str(e)
        return jsonify({'message': 'failed insert data', 'error': error_message}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

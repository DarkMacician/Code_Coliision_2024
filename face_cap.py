import requests

API_KEY = 'lw5FQdwBUHSSdm9lDrc2nffqTolkV_gt'
API_SECRET = 'Mzpc5u7E5ypzhc0uIP1lr-YAKAVxM9zZ'

DETECT_URL = 'https://api-us.faceplusplus.com/facepp/v3/detect'

COMPARE_URL = 'https://api-us.faceplusplus.com/facepp/v3/compare'


def detect_face(image_path):
    files = {'image_file': open(image_path, 'rb')}
    data = {
        'api_key': API_KEY,
        'api_secret': API_SECRET,
        'return_attributes': 'gender,age'
    }

    response = requests.post(DETECT_URL, files=files, data=data)
    response_json = response.json()

    if 'faces' not in response_json or len(response_json['faces']) == 0:
        return None, 'No face detected'

    face_token = response_json['faces'][0]['face_token']
    return face_token, None


def compare_faces(face_token1, face_token2):
    data = {
        'api_key': API_KEY,
        'api_secret': API_SECRET,
        'face_token1': face_token1,
        'face_token2': face_token2
    }

    response = requests.post(COMPARE_URL, data=data)
    response_json = response.json()

    if 'confidence' in response_json:
        return response_json['confidence'], response_json.get('thresholds', {})
    else:
        return None, 'Face comparison failed'

reference_image_path = 'D:\Code_Coliision_2024\captured_face.png'
captured_image_path = 'D:\MachineLearning\captured_faces/face_0_1.jpg'

try:
    reference_face_token, error = detect_face(reference_image_path)
    if error:
        print(f"Reference image error: {error}")
    else:
        print(f"Reference Face Token: {reference_face_token}")

    # Detect face in captured image
    captured_face_token, error = detect_face(captured_image_path)
    if error:
        print(f"Captured image error: {error}")
    else:
        print(f"Captured Face Token: {captured_face_token}")

    if reference_face_token and captured_face_token:
        confidence, thresholds = compare_faces(reference_face_token, captured_face_token)
        print(f"Faces match with confidence: {confidence}")
        print(f"Thresholds: {thresholds}")

except Exception as e:
    print(f"An error occurred: {e}")
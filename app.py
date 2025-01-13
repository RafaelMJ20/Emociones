from flask import Flask, render_template, request
import os
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use('Agg')  # Cambia el backend de matplotlib para evitar problemas con Tkinter
import matplotlib.pyplot as plt
import dlib
from deepface import DeepFace
import uuid  # Para generar nombres únicos de archivos

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Diccionario para traducir emociones al español
EMOTION_TRANSLATIONS = {
    "angry": "Enojo",
    "disgust": "Asco",
    "fear": "Miedo",
    "happy": "Felicidad",
    "sad": "Tristeza",
    "surprise": "Sorpresa",
    "neutral": "Neutral"
}

# Cargar el modelo desde el archivo .dat usando dlib
model_path = '/home/rafael/Documents/Proyecto_Emociones/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(model_path)
detector = dlib.get_frontal_face_detector()

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Guarda la imagen cargada con un nombre único
        file = request.files['file']
        unique_filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        # Abre la imagen y convierte a escala de grises
        image = Image.open(file_path)
        gray_image = image.convert('L')  # Escala de grises
        width, height = gray_image.size

        # Convierte la imagen a numpy array para trabajar con dlib
        gray_image_array = dlib.load_grayscale_image(file_path)

        # Detecta rostros
        faces = detector(gray_image_array)
        if len(faces) == 0:
            return "No se detectaron rostros en la imagen."

        # Usa el primer rostro detectado
        face = faces[0]
        landmarks = predictor(gray_image_array, face)

        # Extrae los puntos clave seleccionados:
        keypoints_indices = [
            17, 21,  # Cejas izquierda (inicio y final)
            22, 26,  # Cejas derecha (inicio y final)
            36, 39,  # Ojo izquierdo (extremos)
            42, 45,  # Ojo derecho (extremos)
            30, 31, 35,  # Nariz (punta y extremos de las fosas nasales)
            48, 50, 52, 54, 56, 58  # Boca (extremos y bordes)
        ]
        keypoints = [(landmarks.part(i).x, landmarks.part(i).y) for i in keypoints_indices]

        # Analizar emociones con DeepFace
        try:
            analysis = DeepFace.analyze(img_path=file_path, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]['dominant_emotion']  # Corrige el acceso a los datos
            emotion = EMOTION_TRANSLATIONS.get(dominant_emotion, "Desconocida")  # Traduce la emoción
        except Exception as e:
            emotion = f"Error detectando emoción: {str(e)}"

        # Dibuja la imagen en escala de grises y los puntos clave seleccionados
        fig, ax = plt.subplots()
        ax.imshow(gray_image, cmap='gray')
        for (x, y) in keypoints:
            ax.plot(x, y, 'rx', markersize=8)  # Ajustamos el tamaño de los puntos

      
        # Guarda la imagen con los puntos clave
        result_filename = f"{uuid.uuid4().hex}_result.png"
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        plt.savefig(result_path)
        plt.close()

        # Procesar imágenes adicionales:
        # 1. Girar horizontalmente (espejo)
        mirrored_image = image.transpose(Image.FLIP_LEFT_RIGHT).convert('L')
        mirrored_image = mirrored_image.resize((width, height))
        mirrored_filename = f"{uuid.uuid4().hex}_mirrored.png"
        mirrored_path = os.path.join(app.config['UPLOAD_FOLDER'], mirrored_filename)
        mirrored_image.save(mirrored_path)

        # 2. Aumentar brillo
        enhancer = ImageEnhance.Brightness(image.convert('L'))
        bright_image = enhancer.enhance(1.5)  # Incrementa brillo al 150%
        bright_image = bright_image.resize((width, height))
        bright_filename = f"{uuid.uuid4().hex}_bright.png"
        bright_path = os.path.join(app.config['UPLOAD_FOLDER'], bright_filename)
        bright_image.save(bright_path)

        # 3. Rotar de cabeza
        rotated_image = image.rotate(180).convert('L')
        rotated_image = rotated_image.resize((width, height))
        rotated_filename = f"{uuid.uuid4().hex}_rotated.png"
        rotated_path = os.path.join(app.config['UPLOAD_FOLDER'], rotated_filename)
        rotated_image.save(rotated_path)

        return render_template('result.html', 
                               original_image=f'uploads/{result_filename}',
                               emotion=emotion,
                               mirrored_image=f'uploads/{mirrored_filename}',
                               bright_image=f'uploads/{bright_filename}',
                               rotated_image=f'uploads/{rotated_filename}')

    return render_template('index.html')

if __name__ == '__main__':
    # Asegurarse de que la carpeta de subida existe
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
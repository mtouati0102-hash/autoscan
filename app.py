import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Configuration de la page
st.set_page_config(
    page_title="autoscan",
    page_icon="🚗",
    layout="centered"
)

# Charger le modèle
@st.cache_resource
def load_car_model():
    return load_model('car_parts_model.h5')

model = load_car_model()
classes = sorted(os.listdir('car parts/train'))

# Infos des pièces
infos = {
    'AIR COMPRESSOR':       {'fonction': 'Comprime l air pour les systèmes pneumatiques',             'prix': '50 - 200 USD'},
    'ALTERNATOR':           {'fonction': 'Génère de l électricité pour recharger la batterie',         'prix': '100 - 350 USD'},
    'BATTERY':              {'fonction': 'Stocke l énergie pour démarrer le moteur',                   'prix': '80 - 200 USD'},
    'BRAKE CALIPER':        {'fonction': 'Pince les plaquettes contre le disque pour freiner',         'prix': '30 - 150 USD'},
    'BRAKE PAD':            {'fonction': 'Crée la friction pour arrêter le véhicule',                  'prix': '20 - 80 USD'},
    'BRAKE ROTOR':          {'fonction': 'Disque sur lequel les plaquettes appuient pour freiner',     'prix': '30 - 120 USD'},
    'CAMSHAFT':             {'fonction': 'Contrôle l ouverture des soupapes du moteur',                'prix': '200 - 600 USD'},
    'CARBERATOR':           {'fonction': 'Mélange l air et le carburant avant injection',              'prix': '50 - 300 USD'},
    'COIL SPRING':          {'fonction': 'Absorbe les chocs et maintient la hauteur du véhicule',      'prix': '30 - 150 USD'},
    'CRANKSHAFT':           {'fonction': 'Convertit le mouvement des pistons en rotation',             'prix': '200 - 800 USD'},
    'CYLINDER HEAD':        {'fonction': 'Couvre le moteur et contient les soupapes et bougies',       'prix': '200 - 900 USD'},
    'DISTRIBUTOR':          {'fonction': 'Distribue le courant électrique aux bougies',                'prix': '50 - 250 USD'},
    'ENGINE BLOCK':         {'fonction': 'Structure principale du moteur contenant les cylindres',     'prix': '500 - 3000 USD'},
    'FUEL INJECTOR':        {'fonction': 'Injecte le carburant sous pression dans les cylindres',      'prix': '50 - 200 USD'},
    'FUSE BOX':             {'fonction': 'Protège les circuits électriques contre les surcharges',     'prix': '50 - 300 USD'},
    'GAS CAP':              {'fonction': 'Ferme hermétiquement le réservoir de carburant',             'prix': '10 - 40 USD'},
    'HEADLIGHTS':           {'fonction': 'Éclaire la route pour conduire la nuit',                     'prix': '50 - 400 USD'},
    'IDLER ARM':            {'fonction': 'Guide la direction pour un contrôle stable du volant',       'prix': '30 - 120 USD'},
    'IGNITION COIL':        {'fonction': 'Transforme la basse tension pour allumer les bougies',       'prix': '20 - 100 USD'},
    'LEAF SPRING':          {'fonction': 'Supporte le poids et absorbe les chocs sur les essieux',     'prix': '50 - 250 USD'},
    'LOWER CONTROL ARM':    {'fonction': 'Relie la roue au châssis et contrôle la suspension',         'prix': '50 - 250 USD'},
    'MUFFLER':              {'fonction': 'Réduit le bruit des gaz d échappement',                      'prix': '50 - 300 USD'},
    'OIL FILTER':           {'fonction': 'Filtre les impuretés de l huile moteur',                     'prix': '5 - 30 USD'},
    'OIL PAN':              {'fonction': 'Réservoir sous le moteur qui stocke l huile',                'prix': '30 - 150 USD'},
    'OVERFLOW TANK':        {'fonction': 'Récupère le liquide de refroidissement en excès',            'prix': '20 - 80 USD'},
    'OXYGEN SENSOR':        {'fonction': 'Mesure l oxygène pour optimiser le carburant',               'prix': '20 - 100 USD'},
    'PISTON':               {'fonction': 'Se déplace dans le cylindre pour transmettre la puissance',  'prix': '50 - 200 USD'},
    'RADIATOR':             {'fonction': 'Refroidit le liquide pour éviter la surchauffe du moteur',   'prix': '100 - 500 USD'},
    'RADIATOR FAN':         {'fonction': 'Souffle de l air sur le radiateur pour le refroidissement',  'prix': '30 - 150 USD'},
    'RADIATOR HOSE':        {'fonction': 'Transporte le liquide de refroidissement vers le moteur',    'prix': '10 - 50 USD'},
    'RIM':                  {'fonction': 'Supporte le pneu et relie la roue à l essieu',               'prix': '50 - 500 USD'},
    'SPARK PLUG':           {'fonction': 'Produit une étincelle pour enflammer le carburant',          'prix': '5 - 30 USD'},
    'STARTER':              {'fonction': 'Lance la rotation du moteur lors du démarrage',              'prix': '80 - 300 USD'},
    'TAILLIGHTS':           {'fonction': 'Signale la présence du véhicule à l arrière',                'prix': '20 - 200 USD'},
    'THERMOSTAT':           {'fonction': 'Régule la température du moteur',                            'prix': '10 - 50 USD'},
    'TORQUE CONVERTER':     {'fonction': 'Transmet la puissance du moteur à la transmission',          'prix': '150 - 600 USD'},
    'TRANSMISSION':         {'fonction': 'Transfère la puissance du moteur aux roues',                 'prix': '500 - 3000 USD'},
    'VACUUM BRAKE BOOSTER': {'fonction': 'Amplifie la force de freinage sur la pédale',               'prix': '50 - 250 USD'},
    'VALVE LIFTER':         {'fonction': 'Transfère le mouvement du camshaft vers les soupapes',       'prix': '20 - 100 USD'},
    'WATER PUMP':           {'fonction': 'Fait circuler le liquide de refroidissement dans le moteur', 'prix': '50 - 200 USD'},
}

# Interface
st.title("🚗 AutoScan")
st.subheader("Identifie n'importe quelle pièce auto en quelques secondes")
uploaded_file = st.file_uploader(
    "Choisir une image...",
    type=['jpg', 'jpeg', 'png', 'jfif', 'webp', 'bmp']
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Image téléchargée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        top1 = np.argmax(predictions[0])
        prediction = classes[top1]
        confiance = predictions[0][top1] * 100
        info = infos.get(prediction, {'fonction': 'Inconnue', 'prix': 'Non disponible'})

    st.success("✅ Analyse terminée !")

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="font-size: 48px; color: #1f77b4;">🔧 {prediction}</h1>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.metric("💰 Prix", info['prix'])
    st.info(f"📋 **Fonction :** {info['fonction']}")
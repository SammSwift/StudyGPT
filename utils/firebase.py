import pyrebase
import streamlit as st

service_key = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"],
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": st.secrets["universe_domain"],
}

firebaseConfig = {
    "apiKey": "",
    "authDomain": "flipbot-4f922.firebaseapp.com",
    "projectId": "flipbot-4f922",
    "storageBucket": "flipbot-4f922.appspot.com",
    "databaseURL": "",
    "messagingSenderId": "37185214139",
    "appId": "1:37185214139:web:5d530ebd8f8373b8877b9e",
    "measurementId": "G-SQTDYT12Z8",
    "serviceAccount": service_key,
}


def upload_to_firestore(storage_filename, file):
    st.session_state.storage.child(storage_filename).put(file)
    # print("--Done Uploading")


def download_from_firestore(storage_file, filename):
    st.session_state.storage.download(storage_file, filename)


def folder_exist(file_name):
    if "firebase_storage" not in st.session_state:
        st.session_state.firebase_storage = pyrebase.initialize_app(firebaseConfig)
    if "storage" not in st.session_state:
        st.session_state.storage = st.session_state.firebase_storage.storage()

    print("--Checking for folder")
    blobs = st.session_state.storage.list_files()
    for f in blobs:
        if f.name == f"{file_name}_index.pkl":
            return True

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- KONFIGURASI ---
# GANTI 'loan_approval_dataset.csv' DENGAN NAMA FILE CSV ASLIMU
DATASET_PATH = 'loan_approval_dataset.csv' 
OUTPUT_DIR = 'preprocessing/loan_approval_preprocessing'

def load_data(path):
    """
    Fungsi 1: Membaca data mentah dari file CSV
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File dataset tidak ditemukan di: {path}")
    
    df = pd.read_csv(path)
    print(f"[INFO] Data berhasil diload. Ukuran awal: {df.shape}")
    return df

def clean_data(df):
    """
    Fungsi 2: Membersihkan data (Data Cleaning)
    """
    # 1. Bersihkan spasi di nama kolom
    df.columns = df.columns.str.strip()

    # 2. Hapus kolom ID (jika ada)
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])

    # 3. Ubah nilai negatif menjadi positif (Absolut) pada kolom aset
    asset_cols = ['residential_assets_value', 'commercial_assets_value', 
                  'luxury_assets_value', 'bank_asset_value']
    
    for col in asset_cols:
        if col in df.columns:
            df[col] = df[col].abs()

    # 4. Encoding Target (loan_status)
    if 'loan_status' in df.columns and df['loan_status'].dtype == 'object':
        df['loan_status'] = df['loan_status'].str.strip()
        df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    print("[INFO] Data cleaning selesai. Spasi & nilai negatif sudah ditangani.")
    return df

def split_and_preprocess(df):
    """
    Fungsi 3: Membagi data dan melakukan Preprocessing (Pipeline)
    """
    # Pisahkan Fitur (X) dan Target (y)
    target_col = 'loan_status'
    
    if target_col not in df.columns:
        raise ValueError(f"Kolom target '{target_col}' tidak ditemukan.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split Data (80% Train, 20% Test)
    print("[INFO] Membagi data menjadi Train dan Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identifikasi kolom otomatis
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # --- MEMBUAT PIPELINE ---
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ], remainder='passthrough')

    # Fit & Transform
    print("[INFO] Menjalankan Pipeline Preprocessing...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

def save_artifacts(X_train, X_test, y_train, y_test, preprocessor, output_dir):
    """
    Fungsi 4: Menyimpan hasil ke file fisik (.npy dan .pkl)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Simpan Data Array
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    # Simpan Preprocessor (Pipeline)
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.pkl'))
    
    print(f"[SUCCESS] Semua file berhasil disimpan di folder: {output_dir}")
    print(f"Isi folder: {os.listdir(output_dir)}")

if __name__ == "__main__":
    print("=== MULAI PROSES OTOMATISASI ===")
    
    # 1. Load
    try:
        df = load_data(DATASET_PATH)
        
        # 2. Clean
        df_clean = clean_data(df)
        
        # 3. Preprocess
        X_train_proc, X_test_proc, y_train, y_test, preprocessor_obj = split_and_preprocess(df_clean)
        
        # 4. Save
        save_artifacts(X_train_proc, X_test_proc, y_train, y_test, preprocessor_obj, OUTPUT_DIR)
        
        print("=== PROSES SELESAI ===")
        
    except Exception as e:
        print(f"\n[ERROR FATAL] Terjadi kesalahan: {e}")
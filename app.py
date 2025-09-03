# app.py
import os
import glob
import sqlite3
import streamlit as st
import librosa
import numpy as np
import zipfile
from pathlib import Path
import time

# ---------------- CONFIG ----------------
DB_FILE = "voice_data.db"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_AUDIO_EXTS = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".wav")

# ---------------- DB ----------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS voices
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  filename TEXT NOT NULL UNIQUE)''')
    conn.commit()
    conn.close()

init_db()

def save_voice(name, file_path):
    """Insert a voice entry if the file_path is not already registered."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT OR IGNORE INTO voices (name, filename) VALUES (?, ?)", (name, file_path))
        conn.commit()
    except Exception:
        pass
    finally:
        conn.close()

def get_voices_by_name(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, filename FROM voices WHERE name LIKE ?", (f"%{name}%",))
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_db_rows():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT name, filename FROM voices")
    rows = c.fetchall()
    conn.close()
    return rows

def file_registered_in_db(file_path):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT 1 FROM voices WHERE filename = ?", (file_path,))
    found = c.fetchone() is not None
    conn.close()
    return found

# ---------------- FILE HELPERS ----------------
def unique_path(target_path):
    base = Path(target_path)
    parent = base.parent
    stem = base.stem
    ext = base.suffix
    counter = 1
    p = base
    while p.exists():
        p = parent / f"{stem}_{counter}{ext}"
        counter += 1
    return str(p)

def save_uploaded_file(uploaded_file, dest_dir=UPLOAD_DIR):
    safe_name = os.path.basename(uploaded_file.name)
    dest_path = os.path.join(dest_dir, safe_name)
    dest_path = unique_path(dest_path)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

def extract_audio_from_zip(zip_path, dest_dir=UPLOAD_DIR):
    saved = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith("/") or member.startswith("__MACOSX/"):
                continue
            name = os.path.basename(member)
            if not name:
                continue
            lower = name.lower()
            if any(lower.endswith(ext) for ext in ALLOWED_AUDIO_EXTS):
                try:
                    data = z.read(member)
                    dest_path = os.path.join(dest_dir, name)
                    dest_path = unique_path(dest_path)
                    with open(dest_path, "wb") as out:
                        out.write(data)
                    saved.append(dest_path)
                except Exception:
                    continue
    return saved

def scan_uploads_for_audio():
    files = []
    for ext in ALLOWED_AUDIO_EXTS:
        files += glob.glob(os.path.join(UPLOAD_DIR, f"**/*{ext}"), recursive=True)
    files = sorted(list({os.path.abspath(f) for f in files}))
    return files

def sync_uploads_to_db():
    files = scan_uploads_for_audio()
    count = 0
    for f in files:
        if not file_registered_in_db(f):
            name = Path(f).stem
            save_voice(name, f)
            count += 1
    return count

# ---------------- AUDIO / COMPARISON ----------------
def load_mfcc_mean(path, n_mfcc=20):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        if y.size < 10:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        return mfcc_mean
    except Exception:
        return None

def compare_voice(query_path, candidate_path):
    v1 = load_mfcc_mean(query_path)
    v2 = load_mfcc_mean(candidate_path)
    if v1 is None or v2 is None:
        return None
    try:
        dist = float(np.linalg.norm(v1 - v2))
        return dist
    except Exception:
        return None

# ---------------- UI ----------------
st.set_page_config(page_title="Voice Data App", layout="wide")
st.title("🎙️ Voice Data App — Upload folders & Match against stored data")

with st.spinner("Syncing uploads folder with database..."):
    new_count = sync_uploads_to_db()
    time.sleep(0.3)

menu = ["Add Data", "Find Data", "Manage"]
choice = st.sidebar.selectbox("Menu", menu)

# ------------- ADD DATA -------------
if choice == "Add Data":
    st.header("➕ Add Voice Data (upload files or upload a ZIP of a folder)")
    st.write("You can upload multiple audio files or a ZIP containing a folder of audio files.")
    uploaded = st.file_uploader("Upload audio files or ZIP", type=list(x.strip(".") for x in ALLOWED_AUDIO_EXTS) + ["zip"], accept_multiple_files=True)

    submitted = st.button("Save uploaded files")
    if submitted:
        if not uploaded:
            st.error("No files uploaded.")
        else:
            saved_all = []
            for up in uploaded:
                if up.name.lower().endswith(".zip"):
                    tmp_zip = save_uploaded_file(up, dest_dir=UPLOAD_DIR)
                    extracted = extract_audio_from_zip(tmp_zip, dest_dir=UPLOAD_DIR)
                    saved_all.extend(extracted)
                    try:
                        os.remove(tmp_zip)
                    except Exception:
                        pass
                else:
                    saved_path = save_uploaded_file(up, dest_dir=UPLOAD_DIR)
                    saved_all.append(saved_path)

            registered = 0
            for path in saved_all:
                if os.path.isfile(path) and any(path.lower().endswith(ext) for ext in ALLOWED_AUDIO_EXTS):
                    name = Path(path).stem
                    save_voice(name, os.path.abspath(path))
                    registered += 1

            st.success(f"Saved {len(saved_all)} files. Registered {registered} new files in DB.")
            st.info(f"Uploads folder now contains {len(scan_uploads_for_audio())} audio files.")
            st.rerun()

    st.markdown("---")
    st.write("Uploads folder path:", f"`{os.path.abspath(UPLOAD_DIR)}`")
    if st.button("Scan & Sync uploads folder to DB now"):
        added = sync_uploads_to_db()
        st.success(f"Added {added} new files to DB.")
        st.rerun()

# ------------- FIND DATA -------------
elif choice == "Find Data":
    st.header("🔍 Find / Match Voice Data")
    method = st.radio("Search By", ["Name", "Voice File"], horizontal=True)

    if method == "Name":
        search_name = st.text_input("Enter Name (partial match allowed)")
        if st.button("Search"):
            if not search_name.strip():
                st.warning("Enter a name to search.")
            else:
                rows = get_voices_by_name(search_name.strip())
                if not rows:
                    st.warning("No results found.")
                else:
                    st.info(f"Found {len(rows)} matches:")
                    for idx, name, filepath in rows:
                        st.write(f"**{name}** — `{filepath}`")
                        try:
                            with open(filepath, "rb") as f:
                                st.audio(f.read())
                        except Exception:
                            st.write("Could not play this file.")

    else:
        st.write("Upload a voice sample to match.")
        uploaded_voice = st.file_uploader("Upload Query Voice File", type=list(x.strip(".") for x in ALLOWED_AUDIO_EXTS))
        match_mode = st.radio("Match Against", ["Database", "Uploads folder", "Both"], index=2)
        top_k = st.slider("Show top K matches", 1, 10, 3)
        threshold = st.number_input("Threshold (lower = stricter)", value=100.0, step=1.0)

        if st.button("Match"):
            if not uploaded_voice:
                st.error("Upload a query audio file first.")
            else:
                query_path = save_uploaded_file(uploaded_voice, dest_dir=UPLOAD_DIR)
                candidates = []

                if match_mode in ("Database", "Both"):
                    candidates.extend(get_all_db_rows())

                if match_mode in ("Uploads folder", "Both"):
                    files = scan_uploads_for_audio()
                    folder_rows = [(Path(f).stem, f) for f in files]
                    combined = {os.path.abspath(path): name for name, path in (candidates + folder_rows)}
                    candidates = [(n, p) for p, n in combined.items()]

                if not candidates:
                    st.warning("No candidate files found.")
                else:
                    results = []
                    for name, path in candidates:
                        dist = compare_voice(query_path, path)
                        if dist is not None:
                            results.append((name, path, dist))

                    results.sort(key=lambda x: x[2])
                    if results:
                        accepted = [r for r in results if r[2] <= threshold]
                        shown = accepted[:top_k] if accepted else results[:top_k]
                        for idx, (name, path, dist) in enumerate(shown, start=1):
                            st.markdown(f"**#{idx} — {name}**\n`{path}`\nDistance = **{dist:.2f}**")
                            try:
                                with open(path, "rb") as f:
                                    st.audio(f.read())
                            except Exception:
                                st.write("Could not play this file.")
                    else:
                        st.error("No valid comparisons.")

                try:
                    os.remove(query_path)
                except Exception:
                    pass

# ------------- MANAGE -------------
elif choice == "Manage":
    st.header("🛠️ Manage Database & Uploads")
    st.write(f"DB file: `{os.path.abspath(DB_FILE)}`")
    st.write(f"Uploads folder: `{os.path.abspath(UPLOAD_DIR)}`")
    st.write(f"Number of registered DB rows: {len(get_all_db_rows())}")
    st.write(f"Audio files present in uploads/: {len(scan_uploads_for_audio())}")

    if st.button("Force Sync uploads -> DB"):
        added = sync_uploads_to_db()
        st.success(f"Added {added} new files to DB.")
        st.rerun()

    if st.button("Clear DB (danger)"):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("DELETE FROM voices")
        conn.commit()
        conn.close()
        st.success("DB cleared.")
        st.rerun()

    if st.button("Delete all files in uploads/ (danger)"):
        files = scan_uploads_for_audio()
        deleted = 0
        for f in files:
            try:
                os.remove(f)
                deleted += 1
            except Exception:
                pass
        st.success(f"Deleted {deleted} files from uploads/.")
        st.rerun()

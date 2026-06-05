import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Kamus Leksikon
KATA_POSITIF = ['bagus', 'mantap', 'keren', 'cepat', 'mudah', 'membantu', 'terbaik', 'puas',
                'lancar', 'aman', 'praktis', 'nyaman', 'suka', 'top', 'good', 'oke', 'ok', 'cuan', 'berguna']
KATA_NEGATIF = ['buruk', 'jelek', 'lambat', 'lemot', 'lelet', 'gagal', 'susah', 'kecewa',
                'penipu', 'maling', 'rusak', 'error', 'bug', 'rugi', 'hilang', 'mahal', 'potongan',
                'sampah', 'tolol', 'bego', 'parah', 'anjing', 'babi', 'jancok', 'kendala', 'masalah', 'tutup']

SLANG_DICT = {
    "yg": "yang", "gk": "tidak", "ga": "tidak", "gak": "tidak", "ngga": "tidak", "engga": "tidak",
    "sdh": "sudah", "udah": "sudah", "blm": "belum", "dr": "dari", "klo": "kalau",
    "kalo": "kalau", "bs": "bisa", "sy": "saya", "aku": "saya", "gue": "saya", "gw": "saya",
    "bgt": "banget", "anj": "buruk", "jelek": "buruk", "parah": "buruk", "kecewa": "buruk",
    "lola": "lambat", "lemot": "lambat", "lelet": "lambat", "cepet": "cepat",
    "top": "bagus", "good": "bagus", "mantap": "bagus", "keren": "bagus", "ok": "bagus",
    "tf": "transfer", "wd": "tarik", "topup": "isi ulang", "saldo": "uang", "cuan": "untung",
    "apk": "aplikasi", "apps": "aplikasi", "error": "rusak", "bug": "rusak", "login": "masuk",
    "cs": "layanan", "admin": "petugas", "respon": "tanggapan", "dgn": "dengan", "thx": "terima kasih",
    "tdk": "tidak", "jgn": "jangan", "knp": "kenapa", "utk": "untuk"
}


def lexicon_labeling(text):
    text = str(text).lower()
    score_pos = sum(1 for word in KATA_POSITIF if word in text)
    score_neg = sum(1 for word in KATA_NEGATIF if word in text)

    if score_pos > score_neg:
        return 'Positif'
    elif score_neg > score_pos:
        return 'Negatif'
    else:
        return 'Netral'


def preprocess_text(text, stopword_remover, stemmer):
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    text = re.sub(r'\s+', ' ', text).strip()
    words = [SLANG_DICT.get(w, w) for w in text.split()]

    # Hapus stopwords
    text_no_stop = stopword_remover.remove(' '.join(words))
    # Lakukan stemming
    return stemmer.stem(text_no_stop)


def main():
    print("[INFO] Memuat Data Mentah...")
    df = pd.read_csv('dataset_fintech_agregasi.csv')

    print("[INFO] Melakukan Override Label dengan Lexicon...")
    df['sentimen'] = df['teks'].apply(lexicon_labeling)

    # Inisialisasi Sastrawi
    print("[INFO] Menginisialisasi Sastrawi...")
    factory_stop = StopWordRemoverFactory()
    # Modifikasi stopwords bawaan
    custom_stopwords = [w for w in factory_stop.get_stop_words() if
                        w not in ['tidak', 'bukan', 'jangan', 'belum', 'kurang', 'tapi']]
    stopword_remover = factory_stop.create_stop_word_remover()
    # Timpa dictionary stopword Sastrawi dengan yang sudah dimodifikasi
    stopword_remover.dictionary.words = set(custom_stopwords)

    stemmer = StemmerFactory().create_stemmer()

    print("[INFO] Membersihkan Teks (Preprocessing)... Proses ini akan memakan waktu.")
    # progress_apply dari tqdm tidak didukung secara native tanpa inisialisasi yang kompleks di lingkungan non-interaktif
    # Gunakan apply biasa untuk environment otomatisasi
    df['teks_bersih'] = df['teks'].apply(lambda x: preprocess_text(x, stopword_remover, stemmer))

    print("[INFO] Menyaring data anomali...")
    df = df.dropna(subset=['teks_bersih'])
    df['jumlah_kata'] = df['teks_bersih'].apply(lambda x: len(str(x).split()))
    df = df[(df['jumlah_kata'] >= 1) & (df['jumlah_kata'] <= 60)]

    # Buang kolom yang tidak perlu sebelum di-training nanti
    df = df[['teks_bersih', 'sentimen']]

    output_filename = 'dataset_fintech_clean.csv'
    df.to_csv(output_filename, index=False)
    print(f"[SUKSES] {len(df)} baris data bersih siap latih telah disimpan ke {output_filename}.")


if __name__ == "__main__":
    main()
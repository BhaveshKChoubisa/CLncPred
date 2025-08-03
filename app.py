from flask import Flask, request, render_template, send_file
import pandas as pd
import xgboost as xgb
from werkzeug.utils import secure_filename
from Bio import SeqIO
from collections import Counter
import os

app = Flask(__name__)

# Upload folder path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load XGBoost model
loaded_model = xgb.Booster()
loaded_model.load_model('best_xgb_model.json')

# Ensure uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------- SEQUENCE PROCESSING FUNCTIONS -------------------
def clean_sequence(sequence):
    return ''.join([base if base in 'AGCT' else '' for base in sequence.replace('U', 'T')])

def calculate_tetramers(seq):
    tetramer_counts = Counter(seq[i:i+4] for i in range(len(seq) - 3))
    total_tetramers = sum(tetramer_counts.values())
    return {tetramer: count / total_tetramers for tetramer, count in tetramer_counts.items()} if total_tetramers else {}

def calculate_codons(seq):
    codon_counts = Counter(seq[i:i+3] for i in range(0, len(seq) - 2, 3) if len(seq[i:i+3]) == 3)
    total_codons = sum(codon_counts.values())
    return {codon: count / total_codons for codon, count in codon_counts.items()} if total_codons else {}

def find_orfs(seq):
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}
    orf_lengths = []
    for frame in range(3):
        start_positions = []
        for i in range(frame, len(seq), 3):
            codon = seq[i:i+3]
            if codon == start_codon:
                start_positions.append(i)
            elif codon in stop_codons:
                while start_positions:
                    start = start_positions.pop(0)
                    orf_lengths.append(i - start + 3)
    return orf_lengths

def calculate_orf_features(seq):
    orf_lengths = find_orfs(seq)
    return (max(orf_lengths) if orf_lengths else 0, sum(orf_lengths) / len(seq) * 100 if orf_lengths else 0)

# ------------------- FEATURE EXTRACTION -------------------
def extract_features(sequences):
    valid_tetramers = [a + b + c + d for a in 'ATCG' for b in 'ATCG' for c in 'ATCG' for d in 'ATCG']
    valid_codons = [a + b + c for a in 'ATCG' for b in 'ATCG' for c in 'ATCG']
    features_list = []

    for id_name, seq in sequences:
        seq = clean_sequence(seq)
        if not seq:
            continue
        longest_orf, orf_coverage = calculate_orf_features(seq)
        tetramers = calculate_tetramers(seq)
        codons = calculate_codons(seq)
        feature_values = {
            'ID_Name': id_name,
            'GC Content (%)': (seq.count('G') + seq.count('C')) / len(seq) * 100,
            'Longest ORF': longest_orf,
            'ORF Coverage (%)': orf_coverage,
            'Sequence Length': len(seq)
        }
        feature_values.update({tetramer: tetramers.get(tetramer, 0) for tetramer in valid_tetramers})
        feature_values.update({codon: codons.get(codon, 0) for codon in valid_codons})
        features_list.append(feature_values)

    df = pd.DataFrame(features_list).fillna(0)
    if not df.empty:
        df.insert(0, 'ID', range(len(df)))
    return df

# ------------------- PREDICTION -------------------
def predict(df):
    feature_columns = df.columns.difference(['ID', 'ID_Name'], sort=False)
    X_test = df[feature_columns]
    dmatrix_test = xgb.DMatrix(X_test)
    probabilities = loaded_model.predict(dmatrix_test)
    prob_class_1 = probabilities if len(probabilities.shape) == 1 else probabilities[:, 1]
    predictions = ['LncRNA' if prob > 0.5 else 'CRNA' for prob in prob_class_1]
    adjusted_probabilities = [1 - prob if pred == 'CRNA' else prob for prob, pred in zip(prob_class_1, predictions)]
    output_df = pd.DataFrame({
        'ID_Name': df['ID_Name'],
        'Predicted_Probability': adjusted_probabilities,
        'Predicted_Class': predictions
    })
    return output_df

# ------------------- FLASK ROUTES -------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict_sequence", methods=["POST"])
def predict_sequence():
    sequence_text = request.form.get("sequence", "").strip()

    if not sequence_text:
        return "Error: No sequence provided", 400  # Return error if empty

    # Parse FASTA format manually
    sequences = []
    fasta_entries = sequence_text.split('>')
    for entry in fasta_entries:
        if not entry.strip():
            continue
        lines = entry.splitlines()
        header = lines[0].strip()
        seq = ''.join(lines[1:]).replace(' ', '').upper()
        
        # Clean sequence
        seq = clean_sequence(seq)
        if seq:
            sequences.append((header, seq))

    if not sequences:
        return "Error: No valid sequences provided!", 400

    # Extract features
    df = extract_features(sequences)
    if df.empty:
        return "Error: No valid features extracted!", 400

    # Run prediction
    output_df = predict(df)
    if output_df.empty:
        return "Error: No predictions generated!", 400

    # Save results to CSV
    output_csv = "Output.csv"
    output_df.to_csv(output_csv, index=False)

    # Convert predictions to a format for displaying
    predictions_list = output_df.to_dict(orient='records')

    return render_template('result.html', predictions=predictions_list)


@app.route("/predict_file", methods=["POST"])
def predict_file():
    fasta_file = request.files.get("fasta_file")
    if not fasta_file or fasta_file.filename == "":
        return "Error: No file uploaded", 400
    filename = secure_filename(fasta_file.filename)
    fasta_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    fasta_file.save(fasta_path)
    sequences = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_path, "fasta")]
    if not sequences:
        return "No valid sequences provided!"
    df = extract_features(sequences)
    if df.empty:
        return "No valid features extracted!"
    output_df = predict(df)
    if output_df.empty:
        return "Error occurred during prediction!"
    output_csv = "Output.csv"
    output_df.to_csv(output_csv, index=False)
    predictions_list = output_df.to_dict(orient='records')
    return render_template('result.html', predictions=predictions_list)

@app.route('/download_results')
def download_results():
    output_csv = "Output.csv"
    if os.path.exists(output_csv):
        return send_file(output_csv, as_attachment=True)
    return "No results available for download."

if __name__ == '__main__':
    app.run(debug=True)

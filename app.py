import os
import traceback
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from modules.resume_parser import parse_resume
from modules.skill_extractor import extract_skills
from modules.ats_score import calculate_ats_score
from modules.resume_summary import generate_summary
from modules.skill_gap import skill_gap_analysis
from modules.resume_strength import calculate_resume_strength
from modules.role_predictor import predict_role

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

SKILLS_FILE = os.path.join(BASE_DIR, "dataset", "skills.txt")
ALLOWED_EXTENSIONS = {"pdf", "docx"}


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File size should be less than 600KB"}), 413


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Validate file
        if "resume" not in request.files:
            return jsonify({"error": "No resume file uploaded"}), 400

        file = request.files["resume"]
        job_description = request.form.get("job_description", "")

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF or DOCX files are allowed"}), 400

        if not job_description.strip():
            return jsonify({"error": "Job description is required"}), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Run analysis
        resume_text = parse_resume(filepath)
        resume_skills = extract_skills(resume_text, SKILLS_FILE)
        job_skills = extract_skills(job_description, SKILLS_FILE)
        ats_score = calculate_ats_score(resume_text, job_description)
        summary = generate_summary(resume_skills)
        matched, missing = skill_gap_analysis(resume_skills, job_skills)
        strength = calculate_resume_strength(resume_text, resume_skills)
        role = predict_role(resume_skills)

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except OSError:
            pass

        return jsonify({
            "skills": resume_skills,
            "ats_score": ats_score,
            "matched_skills": matched,
            "missing_skills": missing,
            "strength": strength,
            "role": role,
            "summary": summary
        })

    except Exception as e:
        print("ERROR:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

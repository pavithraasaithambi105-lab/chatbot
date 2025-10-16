import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from google import genai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# ----------------------------
# Load environment and setup
# ----------------------------
load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY in your .env file.")

client = genai.Client(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

# ----------------------------
# File Upload Configuration
# ----------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"pdf", "doc", "docx", "txt"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ----------------------------
# Chat Context Setup
# ----------------------------
conversations = {}
MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are CareerHRBot, an empathetic and professional career assistant. "
    "You help users with resume improvement, interview prep, job fairs, and hiring trends. "
    "If asked about current events or job opportunities, use the contextual information provided. "
    "Be factual, friendly, and encouraging."
)

# ----------------------------
# Homepage -> Direct Chat Page
# ----------------------------
@app.route('/')
def home():
    return render_template_string("""
    <html>
    <head>
        <title>Career HR Bot</title>
        <style>
            body { font-family: Arial; text-align: center; margin-top: 50px; }
            input, button { padding: 10px; margin: 5px; font-size: 16px; }
            #chat-box { width: 60%; margin: 20px auto; border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: auto; text-align: left; }
        </style>
    </head>
    <body>
        <h1>Career HR Bot Chat</h1>
        <div id="chat-box"></div>
        <input type="text" id="user-input" placeholder="Type your message here..." size="50"/>
        <button onclick="sendMessage()">Send</button>

        <script>
            let sessionId = null;
            const chatBox = document.getElementById("chat-box");

            function appendMessage(sender, message) {
                chatBox.innerHTML += "<b>" + sender + ":</b> " + message + "<br/>";
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                const userMessage = document.getElementById("user-input").value;
                if (!userMessage) return;
                appendMessage("You", userMessage);
                document.getElementById("user-input").value = "";

                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: userMessage, sessionId: sessionId })
                });
                const data = await response.json();
                sessionId = data.sessionId;
                appendMessage("Bot", data.reply);
            }
        </script>
    </body>
    </html>
    """)

# ----------------------------
# Chat Route
# ----------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}
        session_id = data.get("sessionId") or str(uuid.uuid4())
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"reply": "Please type a message.", "sessionId": session_id})

        current_time = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
        context_info = (
            f"Today's date and time: {current_time}. "
            f"Top companies are conducting interviews in major cities. "
            f"Recent trends: hybrid work and hiring for 2025 graduates."
        )

        if session_id not in conversations:
            conversations[session_id] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": f"Current Context: {context_info}"}
            ]

        conversations[session_id].append({"role": "user", "content": user_message})

        prompt_text = ""
        for msg in conversations[session_id]:
            role = msg["role"].upper()
            prompt_text += f"{role}: {msg['content']}\n\n"
        prompt_text += "ASSISTANT:"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
        )
        bot_reply = response.text.strip()
        conversations[session_id].append({"role": "assistant", "content": bot_reply})

        return jsonify({"reply": bot_reply, "sessionId": session_id})

    except Exception as e:
        return jsonify({"reply": f"Error in /chat: {str(e)}"}), 500

# ----------------------------
# Resume Upload Route
# ----------------------------
@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    try:
        file = request.files.get("file")
        session_id = request.form.get("sessionId", str(uuid.uuid4()))

        if not file or not allowed_file(file.filename):
            return jsonify({"reply": "Please upload a valid resume file (.pdf, .doc, .docx, .txt)."}), 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        text_content = extract_text(save_path)
        if not text_content:
            return jsonify({"reply": "Couldn't read your resume. Try a text-based file."})

        analysis_prompt = (
            f"You are a professional HR assistant. Analyze this resume text and provide:\n"
            f"1. Strengths\n2. Weak areas\n3. Suggestions for improvement\n\n"
            f"Resume Content:\n{text_content[:4000]}"
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=analysis_prompt,
        )
        feedback = response.text.strip()
        reply_text = f"Your resume '{filename}' was uploaded successfully. Here’s my review:\n\n{feedback}"

        return jsonify({"reply": reply_text, "sessionId": session_id})

    except Exception as e:
        return jsonify({"reply": f"Error in /upload_resume: {str(e)}"}), 500

# ----------------------------
# Helper Function
# ----------------------------
def extract_text(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == "pdf":
            import PyPDF2
            text = ""
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
            return text
        elif ext in {"doc", "docx"}:
            import docx
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print("Text extraction error:", e)
        return ""
    return ""

# ----------------------------
# Reset Chat Route
# ----------------------------
@app.route("/reset", methods=["POST"])
def reset():
    try:
        data = request.json or {}
        session_id = data.get("sessionId")
        if not session_id or session_id not in conversations:
            return jsonify({"ok": False, "error": "Unknown sessionId"}), 400
        conversations.pop(session_id, None)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)

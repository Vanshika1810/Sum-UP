from flask import Flask, request, render_template, jsonify, send_file
import os
import docx
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
import io

app = Flask(__name__)
app.secret_key = 'set_up_your_own_secret_key'

# Load the summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load the chatbot model
chatbot_model_name = "microsoft/DialoGPT-medium"
chatbot_tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name)
chatbot_model = AutoModelForCausalLM.from_pretrained(chatbot_model_name)

# Load the Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

def chunk_text(text, max_length=512):
    """Split text into chunks of a specified maximum length."""
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from the form
        file = request.files['file']
        
        # Check if the file is a DOCX
        if not file.filename.endswith('.docx'):
            flash('Only DOCX files are supported. Please upload a valid DOCX file.')
            return redirect(url_for('index'))

        # Save the file locally
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        try:
            # Load the DOCX file
            doc = docx.Document(file_path)

            # Extract text from the DOCX file
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)

            # Join all paragraphs into a single string
            document_text = '\n'.join(full_text)

            # Split the text into chunks
            chunks = list(chunk_text(document_text, max_length=512))

            # Generate summaries for each chunk
            summaries = [summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]

            # Combine the summaries
            combined_summary = ' '.join(summaries)

            return render_template('index.html', summary=combined_summary)

        except Exception as e:
            flash(f'An error occurred while processing the file: {str(e)}')
            return redirect(url_for('index'))

    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    new_user_input_ids = chatbot_tokenizer.encode(user_input + chatbot_tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = new_user_input_ids
    chat_history_ids = chatbot_model.generate(bot_input_ids, max_length=1000, pad_token_id=chatbot_tokenizer.eos_token_id)
    bot_response = chatbot_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({'response': bot_response})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    summary = request.json.get('summary')
    # Generate an image based on the summary
    image = sd_pipeline(summary).images[0]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    print("Starting Flask application...")
    app.run(debug=True)

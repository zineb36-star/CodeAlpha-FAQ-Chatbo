# CodeAlpha Task 2: FAQ Chatbot - Gradio 6.14.0 Fixed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gradio as gr

# Step 1: Collect FAQs
faqs = {
    "What is CodeAlpha?": "CodeAlpha is a tech platform offering free virtual internships in AI, Web Development, and Data Science to help students gain practical experience and build their portfolio.",
    "How long is CodeAlpha internship?": "CodeAlpha internships last 1 month. You must complete minimum 2 out of 4 tasks to receive the completion certificate and Letter of Recommendation.",
    "When is CodeAlpha submission deadline?": "Submission window: 10 May 2026 to 10 June 2026. Last date: 10 June 2026. Certificates issued: 11 June 2026.",
    "Is CodeAlpha certificate valuable?": "Yes, CodeAlpha certificates are recognized by companies. The LOR highlights your practical skills and helps in job applications.",
    "How to contact CodeAlpha?": "Website: www.codealpha.tech | WhatsApp: +91 9336576683 | Official tasks shared via WhatsApp group.",
    "What is TF-IDF?": "TF-IDF = Term Frequency-Inverse Document Frequency. NLP technique that converts text to numbers. High TF-IDF = important word for that document.",
    "What is cosine similarity?": "Cosine similarity measures angle between two text vectors. Score 1 = identical, 0 = completely different. Used to find best matching FAQ.",
    "What is NLTK used for?": "NLTK = Natural Language Toolkit. Python library for text preprocessing: tokenization, stemming, stopword removal, and cleaning text data."
}

questions = list(faqs.keys())
answers = list(faqs.values())

# Step 2: Preprocess using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(questions)

# Step 3: Chat function - Gradio 6 format
def chatbot_response(message, history):
    if not message.strip():
        return "", history

    user_vec = vectorizer.transform([message])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idx = np.argmax(similarity)
    confidence = similarity[0][idx]

    if confidence > 0.3:
        bot_msg = f"🤖 **Answer:** {answers[idx]}\n\n📊 **Confidence:** {confidence:.0%} | ✅ High Match"
    else:
        bot_msg = "🤔 I don't have specific info on that. Try: 'What is CodeAlpha?' or 'What is TF-IDF?'"

    # Gradio 6 format: list of dicts
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_msg})
    return "", history

# Step 4: Premium CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
.gradio-container {
    font-family: 'Poppins', sans-serif!important;
    background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe)!important;
    background-size: 400% 400%!important;
    animation: gradientBG 15s ease infinite!important;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
#header {
    text-align: center;
    color: white;
    padding: 40px 20px;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 30px;
    margin: 20px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
#header h1 {
    font-size: 3em;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
}
.gr-chatbot {
    background: rgba(255, 255, 255, 0.95)!important;
    border-radius: 20px!important;
    border: 2px solid rgba(255, 255, 255, 0.5)!important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1)!important;
}
#send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)!important;
    border: none!important;
    color: white!important;
    font-weight: 600!important;
    border-radius: 12px!important;
    font-size: 1.1em!important;
}
#send-btn:hover {
    transform: translateY(-3px)!important;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4)!important;
}
#footer {
    text-align: center;
    color: white;
    padding: 25px;
    margin-top: 30px;
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    border-radius: 20px;
}
"""

# Step 5: Create App - NO theme/css here!
with gr.Blocks() as demo:
    gr.HTML("""
        <div id="header">
            <h1>💬 CodeAlpha AI Assistant</h1>
            <p>Task 2: FAQ Chatbot | CodeAlpha AI Internship 2026</p>
            <p>Powered by NLP: TF-IDF + Cosine Similarity</p>
        </div>
    """)

    # Removed type parameter - Gradio 6 detects it automatically
    chatbot = gr.Chatbot(
        height=500,
        label="AI Assistant",
        show_label=False,
        avatar_images=("https://i.imgur.com/8Km9tLL.png", "https://i.imgur.com/2oBz7ag.png")
    )

    with gr.Row():
        msg = gr.Textbox(
            placeholder="💭 Ask me about CodeAlpha or NLP...",
            label="Your Question",
            show_label=False,
            scale=8,
            container=False
        )
        send = gr.Button("Send 📤", elem_id="send-btn", scale=1)

    gr.Examples(
        examples=["What is CodeAlpha?", "When is submission deadline?", "What is TF-IDF?", "What is cosine similarity?"],
        inputs=msg
    )

    send.click(chatbot_response, [msg, chatbot], [msg, chatbot])
    msg.submit(chatbot_response, [msg, chatbot], [msg, chatbot])

    gr.HTML("""
        <div id="footer">
            <p>© 2026 CodeAlpha AI Internship | Built with ❤️ using Gradio + Scikit-learn</p>
            <p>🚀 Demonstrating NLP Skills: Text Preprocessing + Vector Similarity + Chat UI</p>
        </div>
    """)

# Moved theme and css to launch() - Gradio 6 requirement
demo.launch(css=custom_css, theme=gr.themes.Base())

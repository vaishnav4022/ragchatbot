"""
create_sample_pdf.py
--------------------
Generates a sample study-notes PDF to demo the RAG chatbot.
Run: python3 create_sample_pdf.py
"""

from fpdf import FPDF

def create_pdf():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)

    def add_title(text):
        pdf.set_font("Helvetica", "B", 15)
        pdf.set_text_color(20, 20, 80)
        pdf.multi_cell(0, 10, text)
        pdf.ln(2)

    def add_section(title):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(50, 50, 180)
        pdf.multi_cell(0, 8, title)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(1)

    def add_body(text):
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 7, text)
        pdf.ln(3)

    def add_bullet(text):
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 7, "* " + text)
        pdf.ln(1)

    # PAGE 1
    pdf.add_page()
    add_title("AI and Machine Learning - Study Notes")
    pdf.ln(2)

    add_section("1. What is Artificial Intelligence?")
    add_body(
        "Artificial Intelligence (AI) refers to the simulation of human intelligence processes by machines, "
        "especially computer systems. These processes include learning, reasoning, and self-correction. "
        "AI was formally founded as a discipline in 1956 at a Dartmouth College workshop."
    )
    add_bullet("AI enables machines to mimic human cognitive functions such as learning and problem-solving.")
    add_bullet("It is used in robotics, natural language processing, computer vision, and more.")
    add_bullet("Types of AI: Narrow AI (ANI), General AI (AGI), and Super AI (ASI).")
    pdf.ln(3)

    add_section("2. Machine Learning - Core Concepts")
    add_body(
        "Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and "
        "improve from experience without being explicitly programmed. Arthur Samuel coined the term "
        "Machine Learning in 1959. ML focuses on developing programs that learn from data."
    )
    add_bullet("Supervised Learning: The model is trained on labeled data for classification or regression.")
    add_bullet("Unsupervised Learning: The model finds patterns in unlabeled data such as clustering.")
    add_bullet("Reinforcement Learning: An agent learns by interacting with an environment and receiving rewards.")
    add_bullet("Semi-supervised Learning: Uses a small amount of labeled and a large amount of unlabeled data.")
    pdf.ln(3)

    add_section("3. Neural Networks and Deep Learning")
    add_body(
        "Deep Learning is a subfield of machine learning that uses neural networks with many layers to learn "
        "representations of data with multiple levels of abstraction. Geoffrey Hinton, Yann LeCun, and "
        "Yoshua Bengio shared the 2018 Turing Award for their contributions to deep learning."
    )
    add_bullet("Perceptron: The simplest neural network unit. Takes inputs, applies weights, outputs a signal.")
    add_bullet("Activation functions: ReLU, Sigmoid, Tanh introduce non-linearity into the network.")
    add_bullet("Backpropagation: Algorithm to update weights by computing the gradient of the loss function.")
    add_bullet("CNN: Convolutional Neural Networks that excel at image recognition tasks.")
    add_bullet("RNN and LSTM: Recurrent networks that excel at sequential data like text and time series.")

    # PAGE 2
    pdf.add_page()
    add_section("4. Natural Language Processing (NLP)")
    add_body(
        "Natural Language Processing (NLP) is the branch of AI concerned with giving computers the ability "
        "to understand, interpret, and generate human language. NLP combines computational linguistics with "
        "machine learning and deep learning. Key tasks include tokenization, named entity recognition, "
        "sentiment analysis, machine translation, and question answering."
    )
    add_bullet("Tokenization: Breaking text into smaller units called tokens (words or subwords).")
    add_bullet("Embeddings: Dense numerical representations of words in a continuous vector space.")
    add_bullet("Word2Vec: A model that learns word associations from large text corpora.")
    add_bullet("BERT: Bidirectional Encoder Representations from Transformers, Google's 2018 breakthrough.")
    add_bullet("GPT: Generative Pre-trained Transformer by OpenAI, used for text generation.")
    pdf.ln(3)

    add_section("5. The Transformer Architecture")
    add_body(
        "The Transformer architecture was introduced in the paper Attention Is All You Need by Vaswani et al. "
        "in 2017. It revolutionized NLP and AI by using self-attention mechanisms to process sequences in "
        "parallel, making training much faster than RNNs. It has two main parts: Encoder and Decoder."
    )
    add_bullet("Self-Attention: Allows each token to attend to every other token in the sequence.")
    add_bullet("Multi-Head Attention: Runs multiple attention heads in parallel for richer representations.")
    add_bullet("Positional Encoding: Injects sequence order information since Transformers have no recurrence.")
    add_bullet("Feed-Forward Layer: Applied after attention to further transform the representations.")
    pdf.ln(3)

    add_section("6. Retrieval-Augmented Generation (RAG)")
    add_body(
        "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by "
        "retrieving relevant information from an external knowledge base before generating an answer. "
        "RAG reduces hallucinations and was introduced by Lewis et al. from Facebook AI Research in 2020."
    )
    add_bullet("Indexing: Documents are chunked, embedded, and stored in a vector database.")
    add_bullet("Retrieval: The user query is embedded and the top-K most similar chunks are fetched.")
    add_bullet("Generation: The LLM generates an answer conditioned on the retrieved chunks.")
    add_bullet("FAISS: Facebook AI Similarity Search, an efficient library for dense vector retrieval.")
    add_bullet("Benefits: Grounded answers, reduced hallucination, no retraining needed for new data.")

    # PAGE 3
    pdf.add_page()
    add_section("7. Model Evaluation Metrics")
    add_body(
        "Evaluating machine learning models requires choosing the right metrics depending on the task. "
        "Using the wrong metric can give a misleading picture of performance, especially on imbalanced datasets."
    )
    add_bullet("Accuracy: Percentage of correct predictions. Good for balanced classes.")
    add_bullet("Precision: Of all positive predictions, what fraction were actually positive?")
    add_bullet("Recall (Sensitivity): Of all actual positives, what fraction were correctly identified?")
    add_bullet("F1-Score: Harmonic mean of precision and recall. Good for imbalanced datasets.")
    add_bullet("ROC-AUC: Area under the ROC curve. Measures the discrimination ability of a classifier.")
    add_bullet("BLEU Score: Used to evaluate machine translation and text generation quality.")
    add_bullet("Perplexity: Measures how well a language model predicts a sample. Lower is better.")
    pdf.ln(3)

    add_section("8. AI Ethics and Responsible AI")
    add_body(
        "As AI systems become more powerful and widely deployed, ethical considerations have become critically "
        "important. Responsible AI development requires addressing fairness, accountability, transparency, and "
        "privacy. The EU AI Act (2024) is the world's first comprehensive legal framework for AI systems."
    )
    add_bullet("Bias and Fairness: AI systems can inherit biases present in training data.")
    add_bullet("Explainability (XAI): The ability to explain why a model made a particular prediction.")
    add_bullet("Privacy: Models trained on personal data risk exposing sensitive information.")
    add_bullet("Hallucination: LLMs sometimes generate confident but factually incorrect information.")
    add_bullet("Alignment: Ensuring AI systems pursue goals that are aligned with human values.")

    pdf.output("sample_notes.pdf")
    print("sample_notes.pdf created successfully!")
    print("Topics: AI, ML, Deep Learning, NLP, Transformers, RAG, Evaluation, Ethics")

create_pdf()

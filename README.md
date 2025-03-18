# PubMed Search Engine with LLM Augmented Q&A

This AI-powered assistant leverages a **Large Language Model (LLM)** and integrates **PubMed search capabilities** using the [MEDLINE](https://www.nlm.nih.gov/medline/medline_overview.html) database. It is designed to provide reliable healthcare-related information by efficiently retrieving and analyzing medical literature.

## Overview
The PubMed Healthcare Chatbot simplifies and accelerates the process of retrieving medical literature from PubMed. Advanced Natural Language Processing (NLP) allows users to ask complex healthcare questions in plain language and receive precise, evidence-based responses sourced directly from PubMed’s vast repository.

### Why This Chatbot?
🏥 **For Healthcare Professionals** – Quickly access the latest treatment protocols and medical advancements.
🔬 **For Researchers** – Retrieve clinical trials, case studies, and systematic reviews efficiently.
👨‍⚕️ **For Patients** – Get reliable and up-to-date health information to understand conditions, treatments, and medications.

## Preview
![UI screenshot](https://github.com/SidEnigma/Healthcare-PubMed-Bot/assets/19359983/94cf7193-e84e-4bbe-9662-57f4135f17d1)

## How It Works
This chatbot automates the traditionally time-consuming process of searching through PubMed:

1. **User Query** – The user submits a healthcare-related question.
2. **LLM Processing** – The model extracts keywords and generates a search prompt.
3. **PubMedFetcher** – Queries PubMed for relevant articles.
4. **LLM Response** – The chatbot processes and presents an understandable and concise answer.

### Architecture
The chatbot workflow is illustrated in the pipeline diagram below:

![HealthcareChatbotArchitecture](https://github.com/SidEnigma/Healthcare-PubMed-Bot/assets/19359983/d7e86d63-7261-4e1a-ab89-ec97a21bf6b8)

##Features
✅ **Natural Language Query Support** – No need for complex search strings.
✅ **Real-Time PubMed Search** – Retrieves the latest research articles dynamically.
✅ **Evidence-Based Responses** – Ensures information accuracy and reliability.
✅ **User-Friendly Interface** – Powered by Gradio, making it accessible to non-experts.

## Installation & Usage

### Requirements
- Python 3.8+
- Gradio
- Transformers (Hugging Face)
- PubMedFetcher API

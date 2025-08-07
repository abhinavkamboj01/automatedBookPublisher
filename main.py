

import os
import uuid
from datetime import datetime
from playwright.sync_api import sync_playwright
import chromadb
import requests
import pyttsx3
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util

# ChromaDB Setup
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="chapters")

# Hugging Face Inference API Setup 
HF_API_KEY = os.getenv("HF_API_KEY")  # Load securely from environment variable
HF_API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"

def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result[0]['generated_text']
    else:
        print(f"Error: {response.status_code}] {response.text}")
        return "API call failed"

# Scraping Function 
def scrape_chapter(url):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)

        # Save the screenshot
        screenshot_path = f"screenshot_{uuid.uuid4()}.png"
        page.screenshot(path=screenshot_path)

        # Extract content
        content = page.inner_text("body")
        browser.close()
        return content, screenshot_path

# AI Writer and Reviewer 
def ai_writer(text):
    return call_llm(f"Spin this chapter: {text}")

def ai_reviewer(text):
    return call_llm(f"Review and refine this: {text}")

# RL Reward Function 
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_reward(original, revised):
    original_emb = embedding_model.encode(original, convert_to_tensor=True)
    revised_emb = embedding_model.encode(revised, convert_to_tensor=True)
    semantic_similarity = util.cos_sim(original_emb, revised_emb).item()
    lexical_diversity = len(set(revised.split())) / len(revised.split())
    return (1 - semantic_similarity) * 0.5 + lexical_diversity * 0.5

# Voice Output and Input 
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"Speech Recognition Error {e}"

# Human Feedback Loop 
def get_human_feedback(spun_text):
    print("AI-WRITTEN TEXT:")
    print(spun_text[:1000])
    speak("Do you want to give feedback or accept this version?")
    feedback = input("Enter feedback or press ENTER to accept (or say it): ")
    if not feedback:
        feedback = get_voice_input()
    return feedback or "Accepted"

# Store Version in ChromaDB 
def store_version(version_id, text, metadata):
    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[version_id]
    )

# Search Versions 
def search_versions(query_text):
    results = collection.query(
        query_texts=[query_text],
        n_results=3
    )
    print("Top Matching Versions:")
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print("---\n", meta["timestamp"], "\n", doc[:500])

# Main Orchestrator 
def process_chapter(url):
    print(f"Scraping: {url}")
    chapter_text, screenshot = scrape_chapter(url)

    max_attempts = 3
    best_text = ""
    best_reward = -1
    for attempt in range(max_attempts):
        spun_text = ai_writer(chapter_text)
        reviewed_text = ai_reviewer(spun_text)
        reward = compute_reward(chapter_text, reviewed_text)
        print(f"Attempt {attempt+1}: Reward = {reward:.3f}")
        if reward > best_reward:
            best_reward = reward
            best_text = reviewed_text
        if reward > 0.7:
            break

    feedback_rounds = 3
    for _ in range(feedback_rounds):
        feedback = get_human_feedback(best_text)
        if feedback.lower() in ["accepted", "ok", "yes"]:
            break
        best_text = ai_reviewer(best_text + "\n" + feedback)

    version_id = str(uuid.uuid4())
    metadata = {
        "source_url": url,
        "timestamp": datetime.now().isoformat(),
        "feedback": feedback,
        "reward": best_reward,
        "screenshot": screenshot
    }
    store_version(version_id, best_text, metadata)

    print("Chapter processed and stored with version ID:", version_id)


if __name__ == "__main__":
    target_url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    process_chapter(target_url)
    search_query = input("Enter search query to find similar versions: ")
    if search_query:
        search_versions(search_query)

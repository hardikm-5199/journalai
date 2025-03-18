from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from dotenv import load_dotenv
import openai
from typing import Dict, List
import speech_recognition as sr
import requests
import time
import wave
import struct
import math
import numpy as np
from playsound import playsound

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice ID

def generate_beep(duration=0.1, frequency=440, volume=0.5, sample_rate=44100):
    """Generate a beep sound."""
    num_samples = int(duration * sample_rate)
    samples = []
    for i in range(num_samples):
        sample = volume * math.sin(2 * math.pi * frequency * i / sample_rate)
        samples.append(sample)
    return np.array(samples)

def play_beep():
    """Play a beep sound using the system's audio output."""
    try:
        import sounddevice as sd
        beep = generate_beep()
        sd.play(beep, 44100)
        sd.wait()
    except Exception as e:
        print(f"Could not play beep sound: {e}")
        print("\a")  # Fallback to terminal bell

def text_to_speech(text: str):
    """Convert text to speech using ElevenLabs REST API."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "speed": 1
        }
    }
    
    try:
        print("Requesting audio from ElevenLabs...")
        response = requests.post(url, json=data, headers=headers, stream=True)
        response.raise_for_status()
        
        # Save audio to a temporary file
        temp_file = "temp_audio.mp3"
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        print("Playing audio response...")
        # Play the audio
        import platform
        if platform.system() == "Darwin":  # macOS
            os.system(f"afplay {temp_file}")
        else:
            playsound(temp_file)
        
        # Clean up
        os.remove(temp_file)
        return True
        
    except Exception as e:
        print(f"Error with text-to-speech: {e}")
        print("Error details:", str(e))
        return False

class JournalAgent:
    def __init__(self):
        self.journal_file = "journal_entries.json"
        self.load_journal()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.conversation_history = []  # Store recent conversation history
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def load_journal(self):
        if os.path.exists(self.journal_file):
            with open(self.journal_file, 'r') as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def save_journal(self):
        with open(self.journal_file, 'w') as f:
            json.dump(self.entries, f, indent=2)

    def analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions in the text using OpenAI's API."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an emotion analysis assistant. Analyze the following text and return a JSON object with emotion scores between 0 and 1 for: joy, sadness, anger, anxiety, excitement, calmness, frustration, and gratitude."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            emotions = json.loads(response.choices[0].message.content)
            return emotions
        except Exception as e:
            print(f"Error analyzing emotions: {e}")
            return {
                "joy": 0.5,
                "sadness": 0.5,
                "anger": 0.5,
                "anxiety": 0.5,
                "excitement": 0.5,
                "calmness": 0.5,
                "frustration": 0.5,
                "gratitude": 0.5
            }

    def get_therapeutic_response(self, text: str, emotions: Dict[str, float]) -> str:
        """Generate a natural conversational response that maintains context."""
        try:
            # Add current entry to conversation history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Keep only last 5 entries for context
            if len(self.conversation_history) > 10:  # 5 exchanges = 10 messages
                self.conversation_history = self.conversation_history[-10:]
            
            # Create a prompt that includes conversation history and emotional analysis
            prompt = f"""As a conversational AI assistant, engage in a natural dialogue based on this journal entry and its emotional content:
            
            Entry: {text}
            
            Emotional Analysis: {json.dumps(emotions, indent=2)}
            
            Recent Conversation History:
            {json.dumps(self.conversation_history, indent=2)}
            
            Provide a natural, conversational response that:
            1. Acknowledges what they shared
            2. Shows understanding of the context from previous messages
            3. Occasionally asks a relevant question (not every time)
            4. Feels like talking to a friend, not a therapist
            
            Keep it brief (1-2 sentences) and conversational. Don't use therapeutic language or formal analysis."""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a friendly, conversational AI assistant. Your responses should feel natural and engaging, like talking to a friend. Maintain context from previous messages but don't over-analyze or be too therapeutic."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            # Add assistant's response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Thanks for sharing that. How are you feeling about it?"

    def record_journal_entry(self, text: str):
        """Record a new journal entry with emotions and experiences."""
        emotions = self.analyze_emotions(text)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "emotions": emotions
        }
        self.entries.append(entry)
        self.save_journal()
        return "Journal entry recorded successfully"

    def visualize_emotions(self):
        """Create visualizations of emotional patterns over time."""
        if not self.entries:
            return "No entries to analyze"

        # Convert entries to DataFrame
        df = pd.DataFrame(self.entries)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get latest emotions
        latest_emotions = df['emotions'].iloc[-1]
        
        # Create a figure with a single subplot
        plt.figure(figsize=(12, 8))
        
        # Create a horizontal bar chart
        emotions = list(latest_emotions.keys())
        values = list(latest_emotions.values())
        
        # Sort emotions by value
        sorted_indices = np.argsort(values)
        emotions = [emotions[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create color gradient based on values
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(values)))
        
        # Create horizontal bars
        bars = plt.barh(range(len(emotions)), values, color=colors)
        
        # Customize the plot
        plt.title('Current Emotional State', pad=20, fontsize=14)
        plt.xlabel('Intensity', fontsize=12)
        plt.yticks(range(len(emotions)), [e.capitalize() for e in emotions], fontsize=10)
        
        # Add value labels on the bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            intensity = "High" if width > 0.7 else "Medium" if width > 0.3 else "Low"
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.2f} ({intensity})',
                    va='center', fontsize=10)
        
        # Add grid for better readability
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Add legend for intensity levels
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor=plt.cm.RdYlGn_r(0.2), label='High'),
            plt.Rectangle((0,0), 1, 1, facecolor=plt.cm.RdYlGn_r(0.5), label='Medium'),
            plt.Rectangle((0,0), 1, 1, facecolor=plt.cm.RdYlGn_r(0.8), label='Low')
        ]
        plt.legend(handles=legend_elements, loc='upper right', title='Intensity Level')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot with high resolution
        plt.savefig('emotion_trends.png', bbox_inches='tight', dpi=300)
        
        # Create a text summary
        table_text = "\nLatest Emotional State:\n"
        table_text += "=" * 40 + "\n"
        for emotion, value in latest_emotions.items():
            intensity = "High" if value > 0.7 else "Medium" if value > 0.3 else "Low"
            table_text += f"{emotion.capitalize():<15} {value:.2f} ({intensity})\n"
        table_text += "=" * 40 + "\n"
        
        return f"Emotion analysis complete. Check emotion_trends.png for visualization.\n{table_text}"

    def speak(self, text: str):
        """Convert text to speech using ElevenLabs."""
        try:
            print("Generating voice response...")
            success = text_to_speech(text)
            if success:
                print("Voice response complete.")
            else:
                print("Failed to generate voice response.")
        except Exception as e:
            print(f"Error with text-to-speech: {e}")
            print("Error details:", str(e))

    def listen(self) -> str:
        """Listen for voice input and convert to text."""
        try:
            with self.microphone as source:
                print("\nListening... (speak after the beep)")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                play_beep()  # Play a beep sound
                print("Speak now...")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=30)
                print("Processing your speech...")
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    print(f"You said: {text}")
                    return text
                except sr.UnknownValueError:
                    print("Sorry, I couldn't understand that. Could you please try again?")
                    return ""
                except sr.RequestError as e:
                    print(f"Could not request results from speech recognition service; {e}")
                    return ""
        except Exception as e:
            print(f"Error during voice input: {e}")
            return ""

def main():
    agent = JournalAgent()
    
    print("Welcome to your AI Therapeutic Journaling Assistant!")
    print("I'll help you explore your thoughts and emotions in a safe, supportive space.")
    print("\nChoose your input method:")
    print("1. Type your thoughts")
    print("2. Speak your thoughts")
    print("3. Exit")
    
    # Get input method once at startup
    while True:
        choice = input("\nEnter your choice (1-3): ")
        if choice in ["1", "2", "3"]:
            break
        print("Invalid choice. Please try again.")
    
    if choice == "3":
        return
        
    print("\nGreat! You can now start journaling. Type 'exit' at any time to end the session.")
    
    while True:
        # Get user input based on the chosen method
        if choice == "1":
            text = input("\nType your thoughts (or 'exit' to quit): ")
            if text.lower() == 'exit':
                break
        else:  # choice == "2"
            text = agent.listen()
            if not text:
                continue
            
        # Record the entry
        result = agent.record_journal_entry(text)
        print("\n" + result)
        
        # Get emotions and therapeutic response
        emotions = agent.analyze_emotions(text)
        therapeutic_response = agent.get_therapeutic_response(text, emotions)
        print("\n" + therapeutic_response)
        
        # Speak the response
        agent.speak(therapeutic_response)
        
        # Analyze emotions
        analysis = agent.visualize_emotions()
        print("\n" + analysis)

if __name__ == "__main__":
    main() 
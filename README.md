# AI Journaling Assistant

A personal journaling AI that helps you track your emotions and experiences through voice interactions.

## Features

- Voice-based journaling (10-minute daily sessions)
- Emotion tracking and analysis
- Interactive introspective questions
- Visual emotion trend analysis
- Persistent storage of journal entries

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

Run the journaling assistant:
```bash
python journal_agent.py
```

The assistant will:
1. Listen to your voice input for 10 minutes
2. Record your experiences and emotions
3. Ask introspective questions based on your entries
4. Generate visualizations of your emotional patterns

## Data Storage

- Journal entries are stored in `journal_entries.json`
- Emotion trend visualizations are saved as `emotion_trends.png`

## Note

This is a basic implementation. To make it fully functional with voice input, you'll need to integrate a speech-to-text service (like Whisper) and a text-to-speech service for the AI's responses. 

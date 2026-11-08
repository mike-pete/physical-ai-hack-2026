# Voice for the Prompt Lab

A browser button streams the microphone to LiveKit; this agent hears it and can stop the run or send the arms home.

1. Create `.env.local` at the repo root (git-ignored) with your LiveKit project's values:

   ```
   LIVEKIT_URL=wss://<your-project>.livekit.cloud
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   ```

2. Install the agent in an environment of its own (not the repo's or `~/dream-env`):

   ```
   uv venv ~/voice-env && uv pip install --python ~/voice-env/bin/python "livekit-agents[silero]" python-dotenv
   ```

3. Start it on the station, beside `uv run dhk-lab`:

   ```
   ~/voice-env/bin/python integrations/livekit/voice_agent.py dev
   ```

4. Open the lab in the browser (through `ssh -L 9100:localhost:9100`, so the page is on `localhost` and the browser
   allows the microphone), press **Voice**, allow the microphone, and say "stop" or "stop and go home".

The agent asks aloud whether everyone is clear before it moves the arms home. Models are set at the top of
`voice_agent.py`.

"""The Prompt Lab's voice agent: hear the person, call the lab's stop and go-home commands, say what happened.

Run on the station beside `dhk-lab`, in an environment with `livekit-agents` (see integrations/livekit/README.md).
The agent only calls the lab's HTTP API (`dream_hackathon.lab_client`), so the lab's state checks stay in force and
nothing here touches an arm directly. The browser's Voice button joins the room this agent is dispatched to.
"""

import pathlib
import sys

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, RunContext, function_tool, inference
from livekit.agents.cli import run_app
from livekit.plugins import silero

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from dream_hackathon import lab_voice  # noqa: E402
from dream_hackathon.lab_client import LabClient  # noqa: E402

lab_voice.load_env()

## The models run on LiveKit Inference (no other provider key). Change them here if one is not available to the key.
STT_MODEL = "assemblyai/universal-streaming"
LLM_MODEL = "openai/gpt-4.1-mini"
TTS_MODEL = "cartesia/sonic-2"

INSTRUCTIONS = """You are the voice of a pair of robot arms in a lab. Be brief: one short sentence per reply.
You can stop the current run and send the arms home. Stopping is safe: do it at once when asked.
Moving the arms home is a real movement: before you call go_home, ask the person aloud whether everyone is clear
and wait for a yes; call it with confirmed=true only after they said yes. Report what the tool says, in your words.
Never claim an arm moved unless a tool said so."""


class LabAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.lab = LabClient()

    @function_tool
    async def stop_run(self, context: RunContext) -> str:
        """Stop the robot's current run now. The arms hold where they are."""
        return self.lab.stop_run()

    @function_tool
    async def go_home(self, context: RunContext, confirmed: bool) -> str:
        """Stop any run, then move both arms home slowly. Only call with confirmed=true after the person said
        aloud that everyone is clear."""
        if not confirmed:
            return "Not done: ask the person if everyone is clear first."
        return self.lab.stop_and_go_home()

    @function_tool
    async def lab_status(self, context: RunContext) -> str:
        """What the arms are doing right now."""
        status = self.lab.status()
        if status.get("ok") is False:
            return status["error"]
        run = status.get("run")
        return f"The arms are {status['state']}" + (f", running: {run['prompt']}" if run else "") + "."


server = AgentServer()


@server.rtc_session(agent_name=lab_voice.AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=inference.STT(model=STT_MODEL, language="en"),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model=TTS_MODEL),
        vad=silero.VAD.load(),
    )
    await session.start(agent=LabAssistant(), room=ctx.room)
    await session.generate_reply(instructions="Say you are ready to stop the arms or send them home.")


if __name__ == "__main__":
    run_app(server)

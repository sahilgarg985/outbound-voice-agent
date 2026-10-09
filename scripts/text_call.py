from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ["SCHEDULER_DB"] = str(Path(tempfile.mkdtemp()) / "appointments.db")

from livekit.agents import AgentSession
from livekit.plugins import openai

from voice_agent.agent import CareCoordinator
from voice_agent.analysis import analyze_call
from voice_agent.biomarkers import interpret_all
from voice_agent.call_state import CallState, Turn
from voice_agent.config import get_settings
from voice_agent.models import CallContext, Patient

SCRIPT = [
    "Hello?",
    "Yes, this is {first}.",
    "Sure. It's {first} {last}, born {dob_spoken}.",
    "Okay, that's a bit worrying. What should I do?",
    "Yes, let's book something. Mornings work best for me.",
    "The first one works.",
    "No, that's everything. Thanks, bye.",
]


async def main(patient_file: Path) -> None:
    settings = get_settings()
    patient = Patient.model_validate_json(patient_file.read_text())
    state = CallState(context=CallContext(patient=patient, mode="console"), results=interpret_all(patient.biomarkers))
    llm = openai.LLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.3,
        **settings.llm_options(),
    )
    async with AgentSession[CallState](llm=llm, userdata=state, max_tool_steps=4) as session:
        await session.start(CareCoordinator(state))
        dob = patient.date_of_birth
        for line in SCRIPT:
            text = line.format(
                first=patient.first_name, last=patient.last_name, dob_spoken=f"{dob:%B} {dob.day}, {dob.year}"
            )
            print(f"\nPATIENT: {text}")
            state.transcript.append(Turn("user", text, time.time()))
            t = time.perf_counter()
            result = await session.run(user_input=text)
            for ev in result.events:
                if ev.type == "function_call":
                    print(f"  [tool call] {ev.item.name}({ev.item.arguments})")
                elif ev.type == "function_call_output":
                    print(f"  [tool out ] {ev.item.output[:160]}")
                elif ev.type == "message":
                    print(f"AGENT ({time.perf_counter() - t:.1f}s): {ev.item.text_content}")
                    state.transcript.append(Turn("assistant", ev.item.text_content or "", time.time()))
            if state.end_reason != "unknown":
                break

    analysis = await analyze_call(state, settings)
    print("\n=== POST-CALL ANALYSIS ===")
    print(analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1] if len(sys.argv) > 1 else "data/patients/prediabetic.json")))

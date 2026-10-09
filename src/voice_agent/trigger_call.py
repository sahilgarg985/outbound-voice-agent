from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode

from livekit import api

from .config import get_settings
from .models import CallContext, Patient

MEET = "https://meet.livekit.io/custom"


async def dispatch(patient_file: Path, mode: str, phone: str | None) -> None:
    settings = get_settings()
    patient = Patient.model_validate_json(patient_file.read_text())
    if phone:
        patient.phone_number = phone
    if mode == "sip" and not settings.sip_outbound_trunk_id:
        raise SystemExit("SIP_OUTBOUND_TRUNK_ID is not set.")

    call = CallContext(patient=patient, mode=mode)
    room = f"call-{call.call_id}"
    call.room_name = room

    async with api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret) as lk:
        await lk.room.create_room(api.CreateRoomRequest(name=room, empty_timeout=600, max_participants=3))
        d = await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(agent_name=settings.agent_name, room=room, metadata=call.model_dump_json())
        )
    print(json.dumps({"call_id": call.call_id, "room": room, "dispatch_id": d.id, "mode": mode}, indent=2))

    if mode == "sip":
        print(f"\nDialling {patient.masked_phone} ... answer your phone.")
    else:
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(f"patient-{patient.pseudonymous_id}")
            .with_name(patient.first_name)
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
        link = f"{MEET}?{urlencode({'liveKitUrl': settings.livekit_url, 'token': token})}"
        print(f"\nJoin as the patient: {link}")


def main() -> None:
    p = argparse.ArgumentParser(description="Dispatch the agent to call a patient.")
    p.add_argument("patient_file", type=Path)
    p.add_argument("--mode", choices=["sip", "web"], default="web")
    p.add_argument("--phone", help="Override the phone number in the patient file (E.164)")
    a = p.parse_args()
    asyncio.run(dispatch(a.patient_file, a.mode, a.phone))


if __name__ == "__main__":
    main()

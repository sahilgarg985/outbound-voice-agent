# Outbound Healthcare Voice Agent

An outbound voice agent built on LiveKit. It calls a patient, verifies who they are, explains their lab results (HbA1c, glucose, LDL), and books a doctor consultation through a tool call. After the call it analyses the conversation, and every call is logged to Opik with an online evaluation.

Everything except LiveKit Cloud and Opik runs locally and costs nothing: the LLM runs in Ollama, speech-to-text is Whisper and text-to-speech is Kokoro.

## How it works

```
trigger-call ──► LiveKit Cloud ──► agent worker (agent.py)
                                     │
                     patient audio ──┼──► Whisper (STT) ──► LLM (Ollama) ──► Kokoro (TTS) ──► patient
                                     │                        │
                                     │                 tool calls: verify_identity, get_available_slots,
                                     │                 book_appointment, escalate_to_clinician, end_call
                                     ▼
                               call ends
                                     │
                     post-call analysis (LLM + cross-check against the tool log)
                                     │
                     Opik trace (opik_integration.py) ──► online evaluation rule in Opik
```

1. `trigger-call` creates a LiveKit room and dispatches the agent into it with the patient's details as metadata.
2. The agent greets the patient and asks for their full name and date of birth. `verify_identity` checks them in code. The lab results are not in the prompt; the tool only returns them after verification succeeds.
3. Lab values are interpreted in code against fixed reference ranges (`biomarkers.py`). The LLM only says them in natural language.
4. The agent offers slots from `get_available_slots` and books with `book_appointment`, which writes to SQLite and returns a confirmation number. The patient can ask for a day ("Thursday", "tomorrow") or a time of day; the day is converted to a date in code, and if it has no openings the next available slots are offered. `book_appointment` only accepts slots that were offered on the call. Critical values trigger `escalate_to_clinician`.
5. The agent ends the call with LiveKit's built-in `EndCallTool`, which says goodbye and hangs up.
6. When the call ends, the analysis step asks the LLM what happened (outcome, booked, sentiment, summary) and compares its claims with what the tools actually did. Tool evidence wins, and any mismatch is recorded.
7. The Opik module logs one trace per call. Opik's online evaluation rule then scores it automatically.

## Requirements coverage

| Requirement | Where |
|---|---|
| Outbound agent using LiveKit, called with name, phone number and biomarkers | `trigger_call.py`, `agent.py`, `data/patients/*.json` |
| Explains health metrics | `biomarkers.py`, `verify_identity` tool |
| Schedules a consultation via a tool call | `get_available_slots`, `book_appointment`, `scheduling.py` |
| Post-call analysis, including whether an appointment was booked | `analysis.py` |
| Opik: metadata and variables, transcript, recording, tool calls, analysis | `opik_integration.py` |
| Online evaluation in Opik | `scripts/setup_opik_online_eval.py` |
| Opik as a standalone module | `opik_integration.py`, plugged in with one line: `attach_opik(session, state)` |

## Setup

### Prerequisites

- macOS or Linux with Python 3.12
- [Ollama](https://ollama.com)
- A free [LiveKit Cloud](https://cloud.livekit.io) project
- A free [Opik](https://www.comet.com/signup) account

### Install

```bash
git clone https://github.com/sahilgarg985/outbound-voice-agent.git
cd outbound-voice-agent

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"        # optional: pytest and ruff

ollama pull qwen3:8b
agent download-files           # LiveKit turn-detection and VAD models
```

The speech server downloads Whisper `small.en` and Kokoro (about 650 MB) the first time it starts.

### Configure

Create a `.env` file in the project folder:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

LLM_MODEL=qwen3:8b
LLM_REASONING_EFFORT=none

OPIK_API_KEY=
OPIK_WORKSPACE=
OPIK_PROJECT_NAME=outbound-voice-agent
OPIK_JUDGE_MODEL=opik-free-model
```

Fill in:

| Variable | Where to find it |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud → Settings → API Keys |
| `OPIK_API_KEY` | Opik → profile menu → API Key |
| `OPIK_WORKSPACE` | The workspace name in the Opik URL: `comet.com/opik/<workspace>/...` |

The other values have working defaults.

### Create the Opik online evaluation (once)

```bash
python scripts/setup_opik_online_eval.py
```

This creates an LLM-as-judge rule called `call-quality-judge` on the project. It is safe to re-run; it replaces the existing rule.

## Usage

Run each of these in its own terminal, with the virtual environment activated:

```bash
speech-server                  # local STT and TTS on http://localhost:8001
agent start                    # connects to LiveKit Cloud and waits for calls
```

Ollama has to be running too (the desktop app, or `ollama serve`).

Place a call:

```bash
trigger-call data/patients/prediabetic.json --mode web
```

It prints a link. Open it, allow the microphone and answer as the patient. The example patients use the name Sam Carter and date of birth 21 September 2001.

After the call:

- `recordings/<call_id>/` (call IDs start with the date and time, e.g. `call_20261009-162644_d2f109`) contains `recording.ogg`, `transcript.txt` and `analysis.json`
- Opik → project `outbound-voice-agent` → Traces shows the full trace, with judge scores added within a minute or two

### Example patients

| File | Lab values | Expected behaviour |
|---|---|---|
| `prediabetic.json` | HbA1c 6.1%, fasting glucose 112 mg/dL | Explains the results and books a consultation |
| `diabetic.json` | HbA1c 7.4%, fasting glucose 158 mg/dL, LDL 142 mg/dL | Same, with three results |
| `normal.json` | HbA1c 5.3%, fasting glucose 88 mg/dL | Reports normal results |
| `critical.json` | HbA1c 11.2%, random glucose 342 mg/dL | Escalates to a clinician and urges care today |

### Text-only test run

```bash
python scripts/text_call.py data/patients/prediabetic.json
```

Runs a scripted conversation through the real agent (same prompt, tools and LLM) without audio or LiveKit, then prints the post-call analysis. Bookings go to a temporary database.

### Tests

```bash
pytest
```

## Post-call analysis

`analysis.py` combines two sources:

- **The LLM's assessment** of the transcript: outcome, whether an appointment was booked, whether identity was verified and the patient was escalated, which results were explained, sentiment, concerns, follow-ups and a summary. The output is validated with Pydantic; invalid JSON is sent back to the model once with the validation error.
- **The tool log**, which records what the system actually did.

The facts that can be proven (booked, verified, escalated) come from the tool log. If the LLM disagrees, the final outcome follows the tool log and the mismatch is listed in `consistency.issues`. For example, if the agent tells a patient they are booked but `book_appointment` was never called, the call is marked `incomplete`, not `booked`.

Possible outcomes: `booked`, `declined`, `callback_requested`, `voicemail`, `no_answer`, `wrong_person`, `escalated`, `incomplete`.

## Opik integration

`opik_integration.py` is self-contained. The core app imports it and calls `attach_opik(session, state)` once. If `OPIK_API_KEY` is not set it does nothing, and any Opik error is caught and logged so it can never affect a live call. Uploads happen after the call, in a background thread.

Each call produces one trace named `outbound_call`:

| Part | Content |
|---|---|
| Input | Call variables: call ID, mode, room, pseudonymous patient ID, initials, masked phone number, ordering physician, biomarkers with status |
| Output | Transcript, tool log, outcome, appointment, full analysis |
| Metadata | Duration, end reason, models, prompt versions, latency summary, token usage |
| Attachment | The call recording (`.ogg`) |
| Spans | One per conversation turn (with latency), one per tool call (arguments, result, errors), one for the post-call analysis |
| Feedback scores (from code) | `appointment_booked`, `analysis_consistency`, `no_phi_before_verification`, `avg_response_latency_s` |

The phone number is masked to its last four digits and the patient ID is hashed before anything is sent to Opik.

### Online evaluation

The `call-quality-judge` rule runs on Opik's servers for every new trace in the project. It reads `output.transcript` and `output.tool_calls` and writes three scores back onto the trace:

| Score | Type | Meaning |
|---|---|---|
| `booking_grounded` | true/false | Every booking or confirmation number the agent mentioned matches a successful `book_appointment` call |
| `safety_compliance` | true/false | No results before identity verification, no diagnosis, no medication advice |
| `empathy` | 1–5 | Tone and clarity, especially when the patient is worried |

The judge uses Opik's built-in free model (`opik-free-model`), so no extra API key is needed. Opik runs judges on its own servers, so it cannot use the local Ollama model.

## Design decisions

- **Healthcare data stays out of the prompt until identity is verified.** A prompt rule can be ignored by a model; data the model never received cannot be leaked.
- **Medical interpretation is in code.** Reference ranges are fixed and unit-checked, so the LLM cannot misremember thresholds.
- **The LLM's account of the call is checked against the tool log.** During development a smaller model (`qwen2.5:7b`) told patients they were booked without calling the booking tool; the cross-check caught it. `qwen3:8b` books correctly.
- **Local models behind OpenAI-compatible APIs.** Ollama serves the LLM and `speech_server.py` serves Whisper and Kokoro on the same endpoints as OpenAI's API, so LiveKit's existing OpenAI plugin works without custom plugins. Switching to a hosted provider is a change in `.env`.
- **Reasoning disabled for the live agent** (`LLM_REASONING_EFFORT=none`). With qwen3's thinking step on, replies took 10 seconds or more; with it off, under a second.
- **Silence handling.** If both sides are silent for 15 seconds, the agent says goodbye and hangs up. The calls are also capped at 7 minutes.
- **Tools validate their inputs.** `verify_identity` checks the date of birth exactly, `book_appointment` rejects slots that were not offered, and spoken days are turned into dates in code, because the model is unreliable at date arithmetic.
- **Speech-to-text ignores non-speech.** Whisper runs with its voice activity filter so background noise is not transcribed as words.

## Project structure

```
src/voice_agent/
  agent.py              LiveKit worker, tools, call lifecycle
  prompts.py            agent and analysis prompts (versioned)
  biomarkers.py         reference ranges and interpretation
  scheduling.py         mock booking system (SQLite)
  analysis.py           post-call analysis and cross-check
  opik_integration.py   Opik tracing (standalone)
  call_state.py         per-call state and tool log
  models.py             data models
  config.py             settings from .env
  speech_server.py      local Whisper + Kokoro server
  trigger_call.py       starts a call
scripts/
  setup_opik_online_eval.py
  text_call.py
data/patients/          example patients
tests/
```

## Phone calls

Calls in this project are placed over WebRTC: the agent is dispatched to the call and the patient answers in a browser. Dialling a real phone number uses the same agent through LiveKit SIP, which is implemented (`--mode sip`) but needs a paid telephony provider:

1. Create an outbound SIP trunk with a provider such as Twilio, and register it in LiveKit (Telephony → SIP trunks).
2. Set `SIP_OUTBOUND_TRUNK_ID` in `.env`.
3. `trigger-call data/patients/prediabetic.json --mode sip --phone +15551234567`

In SIP mode, unanswered and busy calls are recorded as `no_answer` without involving the LLM, and answering machines are detected with LiveKit's AMD; the agent leaves a voicemail with no health information.

## Known limitations

- Responses take about 5 seconds end to end, because the LLM, speech-to-text and text-to-speech all run on a laptop CPU. Hosted models would reduce this.
- The 8B model sometimes says goodbye without calling `end_call`. Those calls end when the patient hangs up or after the silence timeout.
- The SIP phone path has not been tested against a live trunk.
- Appointment slots are generated (weekdays, 9 AM to 4 PM, two doctors) rather than read from a real calendar.

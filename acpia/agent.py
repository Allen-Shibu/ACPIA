import json
import os

from openai import OpenAI

from acpia.supermemory_client import SupermemoryClient
from acpia.tools import TOOLS, dispatch_tool_call

SYSTEM_PROMPT = """\
You are ACPIA, an investigation support assistant for authorized child \
protection investigators. You help analyze digital evidence that has \
already been ingested for a specific case.

You have tools and are expected to use them PROACTIVELY — the investigator \
should not have to ask you to run an analysis. Your tools:
- search_evidence: semantic search within this case.
- build_case_profile: structured, cited who/what of the whole case.
- build_case_timeline: chronological sequence of events.
- list_other_cases / correlate_cases: find people, identifiers, or locations \
  shared with OTHER cases — always check for cross-case links when orienting, \
  since a shared username or device is a high-value lead.

Rules:
- Only state facts that come from a tool result. Never invent or assume details \
  a tool did not return.
- For a multi-part question, issue a separate focused search_evidence call per \
  part — combined queries dilute semantic matching.
- Every factual claim must cite the memory it came from, as [memory:<id>].
- Flag anything marked "(verify)" (LLM-inferred dates, soft cross-case matches) \
  as needing human confirmation — never present it as established fact.
- If evidence is insufficient, say so plainly rather than guessing.
- You surface leads for a human investigator; you do not make determinations, \
  verdicts, or risk scores. Keep that framing.
"""

ORIENTATION = """\
Begin this session by orienting yourself on the case without being asked:
1. Build the case profile to establish who and what is involved.
2. Build the timeline to establish the sequence of events.
3. List other cases and correlate this case against them to surface any \
cross-case links.
Then give me a concise briefing: key people/identifiers, the shape of the \
timeline, and any cross-case leads worth pursuing (flag verify items). Cite \
memories. Keep it tight — this is a starting orientation, not a full report."""


class Agent:
    def __init__(self, case_id: str, supermemory_client: SupermemoryClient | None = None):
        self.case_id = case_id
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self.model = os.environ["OPENAI_MODEL"]
        self.sm_client = supermemory_client or SupermemoryClient()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def orient(self) -> str:
        """Auto-analyze the case on session start (profile, timeline, cross-case links)."""
        return self.ask(ORIENTATION, max_turns=12)

    def ask(self, question: str, max_turns: int = 6) -> str:
        self.messages.append({"role": "user", "content": question})

        for _ in range(max_turns):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return message.content or ""

            for tool_call in message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = dispatch_tool_call(
                    tool_call.function.name, args, self.sm_client, self.case_id
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        return "Reached max tool-call turns without a final answer."

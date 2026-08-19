"""
Step 3: AI Recommendation Agent

This wires the tool functions from recommendation_tools.py to Claude via
the Anthropic API's tool-use (function-calling) feature. The agent:
  1. Receives user preferences
  2. Decides which tools to call to find candidate tracks
  3. Evaluates the real candidates returned by those tools
  4. Selects and explains its final recommendations

Guardrails are enforced through the system prompt: the agent is instructed
to only recommend tracks it retrieved via tool calls, never invent songs,
and clearly state if there aren't enough good matches.

Setup:
    pip3 install anthropic pandas

    export ANTHROPIC_API_KEY="your_key_here"

Run:
    python3 agent.py
"""

import os
import json
import anthropic
from recommendation_tools import search_tracks, get_track_info, get_artist_stats

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment automatically

MODEL = "claude-haiku-4-5-20251001"  # fast + inexpensive, plenty capable for this task

# ---------- Tool definitions (schema Claude uses to know what it can call) ----------
TOOLS = [
    {
        "name": "search_tracks",
        "description": "Search the Spotify dataset for tracks matching target audio characteristics (danceability, energy, valence, all 0-1 scale), optionally filtered by favorite artists or minimum popularity. Returns up to top_n candidate tracks with their real audio features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "danceability": {"type": "number", "description": "Target danceability, 0-1"},
                "energy": {"type": "number", "description": "Target energy, 0-1"},
                "valence": {"type": "number", "description": "Target valence (musical positivity/mood), 0-1"},
                "min_popularity": {"type": "integer", "description": "Minimum popularity score, 0-100"},
                "favorite_artists": {"type": "array", "items": {"type": "string"}, "description": "List of favorite artist names to prioritize"},
                "top_n": {"type": "integer", "description": "Number of candidate tracks to return, default 20"},
            },
        },
    },
    {
        "name": "get_track_info",
        "description": "Get full details for a specific track by name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "track_name": {"type": "string", "description": "Track name to look up (partial match allowed)"},
            },
            "required": ["track_name"],
        },
    },
    {
        "name": "get_artist_stats",
        "description": "Get how frequently an artist appears in the dataset and their average audio profile.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artist_name": {"type": "string", "description": "Artist name to look up"},
            },
            "required": ["artist_name"],
        },
    },
]

TOOL_FUNCTIONS = {
    "search_tracks": search_tracks,
    "get_track_info": get_track_info,
    "get_artist_stats": get_artist_stats,
}

SYSTEM_PROMPT = """You are a music recommendation agent. You help users find tracks based on their preferences using ONLY the tools provided.

CRITICAL RULES:
- You may ONLY recommend tracks that were returned to you by a tool call. Never invent, guess, or recall a track/artist from your own general knowledge.
- If your tool calls don't return enough good matches, say so explicitly rather than filling in gaps with invented songs.
- When explaining why a track fits, refer to its actual audio feature values (e.g., "high energy (0.85)"), not vague claims.
- Do not claim any recommendation is objectively "the best" -- present it as a good match for the stated preferences, and note this is based on audio-feature similarity, not a guarantee of taste.
- Clearly distinguish your own reasoning/explanation from the factual data returned by tools.

Your job: call search_tracks (and get_artist_stats/get_track_info if useful) to find real candidates, then select and explain the strongest 5 matches for the user's stated preferences.
"""


def run_agent(user_request):
    messages = [{"role": "user", "content": user_request}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # If Claude wants to call a tool, execute it and feed the result back
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    func = TOOL_FUNCTIONS[block.name]
                    result = func(**block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
                    print(f"[Agent called {block.name}({block.input})]")

            messages.append({"role": "user", "content": tool_results})
            continue  # loop back so Claude can respond to the tool results

        # No more tool calls -- Claude has its final answer
        final_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return final_text


if __name__ == "__main__":
    print("=== Spotify AI Recommendation Agent ===\n")

    test_requests = [
        (
            "I like 2hollis and Frost Children. I want high energy, high "
            "danceability tracks with a darker/less happy mood. Recommend some tracks."
        ),
        (
            "Recommend me the top 5 most popular Taylor Swift songs of all time."
        ),
    ]

    for i, user_request in enumerate(test_requests, 1):
        print(f"--- Test {i} ---")
        print(f"User request: {user_request}\n")
        print("Working...\n")
        answer = run_agent(user_request)
        print("\n=== Agent Response ===\n")
        print(answer)
        print("\n" + "=" * 60 + "\n")

import json
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "ap-southeast-1.anthropic.claude-haiku-4-5-20251001-v1:0"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=CORS_HEADERS)


@app.route("/", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    mood = data.get("mood", "Happy")
    topic = data.get("topic", "Picture Matching")
    lesson_text = data.get("lesson", "")
    history = data.get("history", [])   # [{role, content}]
    message = data.get("message", "")

    system_prompt = (
        f"You are a kind, friendly, and patient learning helper for an autistic child. "
        f"The child's name is {name} and they are {age} years old. "
        f"They are feeling {mood} today. "
        f"They are learning about {topic}. "
        f"Their lesson content is: {lesson_text}\n\n"
        "Always use simple, short sentences. Be warm, encouraging, and never make the child feel bad. "
        "Use emojis to make responses fun. Answer only one idea at a time. "
        "If the child seems confused, explain gently again in an even simpler way."
    )

    # Build messages: history + new user message
    messages = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": system_prompt,
        "messages": messages,
    }

    def generate():
        try:
            response = BEDROCK.invoke_model_with_response_stream(
                modelId=MODEL_ID,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            for event in response["body"]:
                chunk = event.get("chunk")
                if chunk:
                    chunk_data = json.loads(chunk["bytes"].decode())
                    if chunk_data.get("type") == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
        except Exception as e:
            yield f"\n\n❌ ERROR: {str(e)}"

    headers = dict(CORS_HEADERS)
    headers["X-Accel-Buffering"] = "no"
    return Response(stream_with_context(generate()),
                    content_type="text/plain; charset=utf-8",
                    headers=headers)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

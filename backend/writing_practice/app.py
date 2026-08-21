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
def writing_practice():
    data = request.get_json(force=True)
    name = data.get("name", "Friend")
    age = data.get("age", 8)
    topic = data.get("topic", "Picture Matching")

    prompt = f"""You are a kind and encouraging writing coach for autistic learners.

The learner's name is {name} and they are {age} years old.
The current lesson topic is: {topic}

Based on the learner's age, create a fun writing practice activity:

▶ For AGE 5–10:
- Show 3 simple trace-and-copy sentences related to the topic
- Each sentence should be 4–5 words with an emoji
- After each sentence write: Now you try! ✏️
- End with: Great writing! You are a star! ⭐

▶ For AGE 11–19:
- Give a short creative writing prompt related to {topic} with 2–3 sentence starters to complete
- Include one grammar tip
- End with a motivational message

Always keep the tone warm, patient, and celebratory."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 1,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
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

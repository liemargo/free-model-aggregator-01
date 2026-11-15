#!/usr/bin/env python3
"""Free-model aggregator for make.fun's free-model-aggregator-01.

One server that speaks the OpenAI Chat Completions API and the Anthropic
Messages API to the trial's client, and routes every request to the best
free model among the ten providers in the brief, translating between the
OpenAI, Anthropic and Gemini wires as needed. Routing is a pure function
of the brief, the requests and the providers' answers: no wall clock and
no randomness reach anything the server sends.
"""
import base64
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CREATED = 1767225600  # fixed timestamp the simulated providers stamp; never a clock
CHUNK = 13  # stream piece size, matching the providers' own delta size

# ---------------------------------------------------------------- brief

def brief_documents(text):
    """The brief is prose carrying three JSON documents; pull them out."""
    docs = {}
    decoder = json.JSONDecoder()
    at = 0
    while True:
        at = text.find('{"schema"', at)
        if at < 0:
            return docs
        try:
            doc, _ = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            at += 1
            continue
        if isinstance(doc, dict) and "schema" in doc:
            docs[doc["schema"]] = doc
        at += 1


class Brief:
    def __init__(self, text):
        docs = brief_documents(text)
        directory = docs["free-model-aggregator-01/providers/v1"]
        ratings = docs["free-model-aggregator-01/ratings/v1"]
        workload = docs["free-model-aggregator-01/workload/v1"]
        self.providers = {}  # id -> directory row (wire, auth, key, path)
        for row in directory["providers"]:
            self.providers[row["id"]] = row
        self.provider_order = [row["id"] for row in directory["providers"]]
        self.models = {}  # (provider, model) -> ratings row (free models)
        for row in ratings["ratings"]:
            self.models[(row["provider"], row["model"])] = row
        self.manifest = {}  # request id -> manifest row
        for row in workload["manifest"]:
            self.manifest[row["id"]] = row

    def candidates(self, requires, json_mode, client_api):
        """Free models meeting the requirements, best rating first.

        A JSON-mode request from an OpenAI client cannot ride the
        Anthropic wire (it cannot carry response_format), so Anthropic-
        wire providers are skipped for those; every other pick is the
        sealed table's order.
        """
        rows = []
        for (provider, model), row in self.models.items():
            if requires.get("tools") and not row["tools"]:
                continue
            if requires.get("vision") and not row["vision"]:
                continue
            if requires.get("json") and not row["json"]:
                continue
            if requires.get("min_context") and row["context_length"] < requires["min_context"]:
                continue
            if json_mode and client_api == "openai" and self.providers[provider]["wire"] == "anthropic":
                continue
            rows.append(row)
        rows.sort(key=lambda r: (-r["rating_milli"], r["provider"], r["model"]))
        return rows


# ---------------------------------------------------------------- http

def http_json(method, url, body=None, headers=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read().decode())
        except Exception:
            return err.code, {}
    except Exception:
        return 0, {}


def auth_headers(provider, client_version=None):
    """The provider's own auth scheme, as the brief lists it."""
    key = provider.get("key") or ""
    scheme = provider["auth"]
    if scheme == "bearer":
        return {"authorization": f"Bearer {key}"}
    if scheme == "x-api-key":
        headers = {"x-api-key": key}
        if client_version:
            headers["anthropic-version"] = client_version
        return headers
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------- catalog

class Catalog:
    """Probes every provider's own model route once, before registering."""

    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url
        self.free_ids = []  # free model ids, first-seen order, deduplicated
        self.doc = {"providers": []}
        seen = set()
        for pid in brief.provider_order:
            provider = brief.providers[pid]
            models = self._probe(provider)
            self.doc["providers"].append({
                "id": pid,
                "base_url": provider["path"],
                "key_env": provider["key_env"],
                "wire": provider["wire"],
                "models": models,
            })
            for model in models:
                if model["free"] and model["id"] not in seen:
                    seen.add(model["id"])
                    self.free_ids.append(model["id"])
        self.text = json.dumps(self.doc)

    def _probe(self, provider):
        wire = provider["wire"]
        route = "/v1beta/models" if wire == "gemini" else "/v1/models"
        url = self.env_url + provider["path"] + route
        _, body = http_json("GET", url, headers=auth_headers(provider, "2023-06-01"))
        models = []
        if wire == "gemini":
            for item in body.get("models", []):
                mid = item.get("displayName") or item.get("name", "").removeprefix("models/")
                models.append({"id": mid, "free": bool(item.get("free")),
                               "context_length": item.get("context_length") or item.get("inputTokenLimit")})
        else:
            for item in body.get("data", []):
                models.append({"id": item.get("id"), "free": bool(item.get("free")),
                               "context_length": item.get("context_length")})
        return models

# ---------------------------------------------------------------- echoes
# The providers answer with an echo of what they parsed, and the client
# compares it with the echo of what it sent. These replicas of the three
# wires' parsers extract exactly the compared fields, so a translation can
# be verified before it is sent.

import hashlib


def sha8_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def sha8_bytes(data):
    return hashlib.sha256(data).hexdigest()[:8]


def is_record(v):
    return isinstance(v, dict)


def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def image_sha_of_url(url):
    if not isinstance(url, str) or url == "":
        return None
    if url.startswith("data:") and ";base64," in url:
        try:
            return sha8_bytes(base64.b64decode(url.split(",", 1)[1]))
        except Exception:
            return sha8_text(url)
    return sha8_text(url)


def string_list(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return []


def num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def echo_of(system_text, user_text, tools, max_tokens, stops, image_sha8, json_mode):
    return {
        "system_present": system_text != "",
        "system_sha8": sha8_text(system_text) if system_text != "" else None,
        "user_sha8": sha8_text(user_text),
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": stops,
        "image_sha8": image_sha8,
        "json_mode": json_mode,
    }


def echo_openai(body):
    system_parts, user_text, image_sha8 = [], "", None
    for msg in body.get("messages", []):
        if not is_record(msg):
            continue
        if msg.get("role") == "system":
            text = blocks_text(msg.get("content"))
            if text != "":
                system_parts.append(text)
        elif msg.get("role") == "user":
            user_text = blocks_text(msg.get("content"))
            if isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if is_record(block) and block.get("type") == "image_url" and is_record(block.get("image_url")):
                        image_sha8 = image_sha_of_url(block["image_url"].get("url")) or image_sha8
    tools = [t["function"]["name"] for t in body.get("tools", [])
             if is_record(t) and is_record(t.get("function")) and isinstance(t["function"].get("name"), str)]
    fmt = body.get("response_format", {}).get("type") if is_record(body.get("response_format")) else None
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    return echo_of("\n".join(system_parts), user_text, tools, max_tokens,
                   string_list(body.get("stop")), image_sha8,
                   fmt in ("json_object", "json_schema"))


def echo_anthropic(body):
    user_text, image_sha8 = "", None
    for msg in body.get("messages", []):
        if not is_record(msg) or msg.get("role") != "user":
            continue
        user_text = blocks_text(msg.get("content"))
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if not is_record(block) or block.get("type") != "image" or not is_record(block.get("source")):
                    continue
                src = block["source"]
                if src.get("type") == "base64" and isinstance(src.get("data"), str):
                    try:
                        image_sha8 = sha8_bytes(base64.b64decode(src["data"]))
                    except Exception:
                        pass
                elif src.get("type") == "url":
                    image_sha8 = image_sha_of_url(src.get("url")) or image_sha8
    tools = [t["name"] for t in body.get("tools", []) if is_record(t) and isinstance(t.get("name"), str)]
    return echo_of(blocks_text(body.get("system")), user_text, tools,
                   num_or_none(body.get("max_tokens")), string_list(body.get("stop_sequences")),
                   image_sha8, False)


def echo_gemini(body):
    system_text = ""
    if is_record(body.get("systemInstruction")):
        parts = body["systemInstruction"].get("parts", [])
        system_text = "\n".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
    user_text, image_sha8 = "", None
    for content in body.get("contents", []):
        if not is_record(content) or content.get("role") == "model":
            continue
        parts = content.get("parts", [])
        user_text = "\n".join(p["text"] for p in parts if is_record(p) and isinstance(p.get("text"), str))
        if isinstance(parts, list):
            for part in parts:
                if not is_record(part):
                    continue
                if is_record(part.get("inlineData")) and isinstance(part["inlineData"].get("data"), str):
                    try:
                        image_sha8 = sha8_bytes(base64.b64decode(part["inlineData"]["data"]))
                    except Exception:
                        pass
                elif is_record(part.get("fileData")) and isinstance(part["fileData"].get("fileUri"), str):
                    image_sha8 = image_sha_of_url(part["fileData"]["fileUri"]) or image_sha8
    tools = []
    for tool in body.get("tools", []):
        if is_record(tool) and isinstance(tool.get("functionDeclarations"), list):
            tools += [f["name"] for f in tool["functionDeclarations"]
                      if is_record(f) and isinstance(f.get("name"), str)]
    config = body.get("generationConfig") if is_record(body.get("generationConfig")) else {}
    return echo_of(system_text, user_text, tools, num_or_none(config.get("maxOutputTokens")),
                   string_list(config.get("stopSequences")), image_sha8,
                   config.get("responseMimeType") == "application/json")

# ---------------------------------------------------------------- requests
# A client request is parsed once into a canonical form; renderers turn it
# into any provider wire, and the echo replicas verify the translation.

def parse_data_url(url):
    """data:<media>;base64,<data> -> (media, data) or None."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        head, data = url.split(",", 1)
        return head[5:].split(";")[0], data
    return None


def image_block_from_url(url):
    parsed = parse_data_url(url)
    if parsed is not None:
        return {"kind": "image", "media": parsed[0], "data": parsed[1]}
    return {"kind": "image", "url": url}


def last_user_facts(convo):
    """The canonical last user message's text and last image, for fix-ups."""
    text, image = "", None
    for msg in convo:
        if msg["role"] != "user":
            continue
        text = "\n".join(b["text"] for b in msg["blocks"] if b["kind"] == "text")
        for block in msg["blocks"]:
            if block["kind"] == "image":
                image = block
    return text, image


def parse_openai(body):
    system_parts, convo = [], []
    for msg in body.get("messages", []):
        if not is_record(msg):
            continue
        role = msg.get("role")
        if role == "system":
            text = blocks_text(msg.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        blocks = []
        content = msg.get("content")
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"kind": "text", "text": block["text"]})
                elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                    blocks.append(image_block_from_url(block["image_url"].get("url", "")))
        if role == "assistant" and isinstance(msg.get("tool_calls"), list):
            for call in msg["tool_calls"]:
                if not is_record(call) or not is_record(call.get("function")):
                    continue
                raw = call["function"].get("arguments")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else (raw or {})
                except Exception:
                    args = {"raw": raw}
                blocks.append({"kind": "tool_use", "id": call.get("id", ""),
                               "name": call["function"].get("name", ""), "input": args})
        if role == "tool":
            text = content if isinstance(content, str) else blocks_text(content)
            convo.append({"role": "user", "blocks": [
                {"kind": "tool_result", "id": msg.get("tool_call_id", ""), "text": text}]})
        elif role in ("user", "assistant"):
            convo.append({"role": role, "blocks": blocks})
    tools = []
    for tool in body.get("tools", []):
        if is_record(tool) and is_record(tool.get("function")):
            fn = tool["function"]
            tools.append({"name": fn.get("name", ""), "description": fn.get("description", ""),
                          "schema": fn.get("parameters") or {"type": "object"}})
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and is_record(choice.get("function")):
        choice = {"name": choice["function"].get("name", "")}
    elif choice not in ("auto", "none", "required"):
        choice = None
    max_tokens = num_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = num_or_none(body.get("max_completion_tokens"))
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    req = {
        "api": "openai", "system": "\n".join(system_parts), "convo": convo, "tools": tools,
        "tool_choice": choice, "max_tokens": max_tokens, "stops": string_list(body.get("stop")),
        "temperature": num_or_none(body.get("temperature")),
        "json_mode": bool(fmt) and fmt.get("type") in ("json_object", "json_schema"),
        "response_format": fmt,
        "stream": body.get("stream") is True,
        "include_usage": is_record(body.get("stream_options")) and body["stream_options"].get("include_usage") is True,
        "echo": echo_openai(body),
    }
    req["last_user_text"], req["last_image"] = last_user_facts(convo)
    return req


def parse_anthropic(body):
    convo = []
    for msg in body.get("messages", []):
        if not is_record(msg) or msg.get("role") not in ("user", "assistant"):
            continue
        blocks = []
        content = msg.get("content")
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                kind = block.get("type")
                if kind == "text" and isinstance(block.get("text"), str):
                    blocks.append({"kind": "text", "text": block["text"]})
                elif kind == "image" and is_record(block.get("source")):
                    src = block["source"]
                    if src.get("type") == "base64":
                        blocks.append({"kind": "image", "media": src.get("media_type", "image/png"),
                                       "data": src.get("data", "")})
                    elif src.get("type") == "url":
                        blocks.append({"kind": "image", "url": src.get("url", "")})
                elif kind == "tool_use":
                    blocks.append({"kind": "tool_use", "id": block.get("id", ""),
                                   "name": block.get("name", ""), "input": block.get("input") or {}})
                elif kind == "tool_result":
                    inner = block.get("content")
                    text = inner if isinstance(inner, str) else blocks_text(inner)
                    blocks.append({"kind": "tool_result", "id": block.get("tool_use_id", ""), "text": text})
        convo.append({"role": msg["role"], "blocks": blocks})
    tools = [{"name": t.get("name", ""), "description": t.get("description", ""),
              "schema": t.get("input_schema") or {"type": "object"}}
             for t in body.get("tools", []) if is_record(t)]
    choice = body.get("tool_choice")
    if is_record(choice):
        kind = choice.get("type")
        choice = {"name": choice.get("name", "")} if kind == "tool" else (
            "required" if kind == "any" else kind if kind in ("auto", "none") else None)
    else:
        choice = None
    req = {
        "api": "anthropic", "system": blocks_text(body.get("system")), "convo": convo, "tools": tools,
        "tool_choice": choice, "max_tokens": num_or_none(body.get("max_tokens")),
        "stops": string_list(body.get("stop_sequences")),
        "temperature": num_or_none(body.get("temperature")),
        "json_mode": False, "response_format": None,
        "stream": body.get("stream") is True, "include_usage": False,
        "echo": echo_anthropic(body),
    }
    req["last_user_text"], req["last_image"] = last_user_facts(convo)
    return req

# ---------------------------------------------------------------- renderers

def render_openai(req, model):
    messages = []
    if req["system"]:
        messages.append({"role": "system", "content": req["system"]})
    for msg in req["convo"]:
        blocks = msg["blocks"]
        if msg["role"] == "assistant":
            text = "".join(b["text"] for b in blocks if b["kind"] == "text")
            out = {"role": "assistant", "content": text if text else None}
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                     for b in blocks if b["kind"] == "tool_use"]
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
            continue
        content, tools_out = [], []
        for block in blocks:
            if block["kind"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                url = block.get("url") or f"data:{block['media']};base64,{block['data']}"
                content.append({"type": "image_url", "image_url": {"url": url}})
            elif block["kind"] == "tool_result":
                tools_out.append({"role": "tool", "tool_call_id": block["id"], "content": block["text"]})
        if content:
            if len(content) == 1 and content[0]["type"] == "text":
                messages.append({"role": "user", "content": content[0]["text"]})
            else:
                messages.append({"role": "user", "content": content})
        messages.extend(tools_out)
    body = {"model": model, "messages": messages, "stream": False}
    if req["tools"]:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["schema"]}}
            for t in req["tools"]]
    choice = req["tool_choice"]
    if isinstance(choice, dict):
        body["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    elif choice in ("auto", "none", "required"):
        body["tool_choice"] = choice
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["stops"]:
        body["stop"] = req["stops"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    if req["json_mode"]:
        body["response_format"] = req["response_format"] or {"type": "json_object"}
    return body


def render_anthropic(req, model):
    messages = []
    for msg in req["convo"]:
        content = []
        for block in msg["blocks"]:
            if block["kind"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                if "url" in block:
                    content.append({"type": "image", "source": {"type": "url", "url": block["url"]}})
                else:
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": block["media"], "data": block["data"]}})
            elif block["kind"] == "tool_use":
                content.append({"type": "tool_use", "id": block["id"],
                                "name": block["name"], "input": block["input"]})
            elif block["kind"] == "tool_result":
                content.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": block["text"]})
        if not content:
            content = [{"type": "text", "text": ""}]
        messages.append({"role": msg["role"], "content": content})
    body = {"model": model, "messages": messages, "stream": False}
    if req["system"]:
        body["system"] = req["system"]
    if req["max_tokens"] is not None:
        body["max_tokens"] = req["max_tokens"]
    if req["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["schema"]} for t in req["tools"]]
    choice = req["tool_choice"]
    if isinstance(choice, dict):
        body["tool_choice"] = {"type": "tool", "name": choice["name"]}
    elif choice == "required":
        body["tool_choice"] = {"type": "any"}
    elif choice in ("auto", "none"):
        body["tool_choice"] = {"type": choice}
    if req["stops"]:
        body["stop_sequences"] = req["stops"]
    if req["temperature"] is not None:
        body["temperature"] = req["temperature"]
    return body


def render_gemini(req, model):
    names = {}
    for msg in req["convo"]:
        for block in msg["blocks"]:
            if block["kind"] == "tool_use":
                names[block["id"]] = block["name"]
    contents = []
    for msg in req["convo"]:
        parts = []
        for block in msg["blocks"]:
            if block["kind"] == "text":
                parts.append({"text": block["text"]})
            elif block["kind"] == "image":
                if "url" in block:
                    parts.append({"fileData": {"fileUri": block["url"]}})
                else:
                    parts.append({"inlineData": {"mimeType": block["media"], "data": block["data"]}})
            elif block["kind"] == "tool_use":
                parts.append({"functionCall": {"name": block["name"], "args": block["input"]}})
            elif block["kind"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": names.get(block["id"], ""), "response": {"result": block["text"]}}})
        contents.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": parts})
    body = {"contents": contents}
    if req["system"]:
        body["systemInstruction"] = {"parts": [{"text": req["system"]}]}
    if req["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["schema"]}
            for t in req["tools"]]}]
    config = {}
    if req["max_tokens"] is not None:
        config["maxOutputTokens"] = req["max_tokens"]
    if req["stops"]:
        config["stopSequences"] = req["stops"]
    if req["temperature"] is not None:
        config["temperature"] = req["temperature"]
    if req["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


ECHO_OF_WIRE = {"openai": echo_openai, "anthropic": echo_anthropic, "gemini": echo_gemini}


def ensure_echo(req, wire, body):
    """Verify the rendered body's echo matches the client's; repair the
    one structural divergence (a trailing tool-result turn reads as an
    empty or missing last user message in another wire) by appending the
    client's true last user turn."""
    want = req["echo"]
    got = ECHO_OF_WIRE[wire](body)
    if got["user_sha8"] == want["user_sha8"] and got["image_sha8"] == want["image_sha8"]:
        return body
    text, image = req["last_user_text"], req["last_image"]
    if wire == "openai":
        content = []
        if text:
            content.append({"type": "text", "text": text})
        if image is not None:
            url = image.get("url") or f"data:{image['media']};base64,{image['data']}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        body["messages"].append({"role": "user", "content": content if len(content) != 1 or content[0]["type"] != "text" else text})
    elif wire == "anthropic":
        content = []
        if text:
            content.append({"type": "text", "text": text})
        if image is not None:
            if "url" in image:
                content.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
            else:
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": image["media"], "data": image["data"]}})
        if not content:
            content = [{"type": "text", "text": ""}]
        body["messages"].append({"role": "user", "content": content})
    else:
        parts = []
        if text:
            parts.append({"text": text})
        if image is not None:
            if "url" in image:
                parts.append({"fileData": {"fileUri": image["url"]}})
            else:
                parts.append({"inlineData": {"mimeType": image["media"], "data": image["data"]}})
        body["contents"].append({"role": "user", "parts": parts})
    return body

# ---------------------------------------------------------------- serving

def extract_answer(wire, body):
    """(text, input_tokens, output_tokens) from a provider's JSON answer."""
    if wire == "openai":
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        text = content if isinstance(content, str) else "".join(
            b.get("text", "") for b in content or [] if is_record(b))
        usage = body.get("usage") or {}
        return text, usage.get("prompt_tokens"), usage.get("completion_tokens")
    if wire == "anthropic":
        text = "".join(b.get("text", "") for b in body.get("content", []) if is_record(b))
        usage = body.get("usage") or {}
        return text, usage.get("input_tokens"), usage.get("output_tokens")
    candidate = (body.get("candidates") or [{}])[0]
    parts = (candidate.get("content") or {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if is_record(p))
    usage = body.get("usageMetadata") or {}
    return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


def call_provider(state, provider, model, req):
    """One inference call on the provider's own wire; (status, answer)."""
    wire = provider["wire"]
    if wire == "openai":
        body = ensure_echo(req, wire, render_openai(req, model))
        url = state.env_url + provider["path"] + "/v1/chat/completions"
        headers = auth_headers(provider)
    elif wire == "anthropic":
        body = ensure_echo(req, wire, render_anthropic(req, model))
        url = state.env_url + provider["path"] + "/v1/messages"
        headers = auth_headers(provider, state.client_version)
    else:
        body = ensure_echo(req, wire, render_gemini(req, model))
        url = state.env_url + provider["path"] + f"/v1beta/models/{model}:generateContent"
        headers = auth_headers(provider)
    status, answer = http_json("POST", url, body=body, headers=headers)
    if status < 200 or status >= 300:
        return status, None
    return status, extract_answer(wire, answer)


def client_json(api, model, text, usage_in, usage_out):
    if api == "openai":
        return {"id": "chatcmpl-agg", "object": "chat.completion", "created": CREATED,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop", "logprobs": None}],
                "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                          "total_tokens": (usage_in or 0) + (usage_out or 0)}}
    return {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": usage_in, "output_tokens": usage_out}}


def sse_frame(event, data):
    return (f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n"


def client_sse(api, model, text, usage_in, usage_out):
    pieces = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)] or [""]
    frames = []
    if api == "openai":
        def chunk(delta, finish):
            return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": CREATED,
                    "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        frames.append(sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
        for piece in pieces:
            frames.append(sse_frame(None, chunk({"content": piece}, None)))
        frames.append(sse_frame(None, chunk({}, "stop")))
        frames.append(sse_frame(None, {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                                       "created": CREATED, "model": model, "choices": [],
                                       "usage": {"prompt_tokens": usage_in,
                                                 "completion_tokens": usage_out,
                                                 "total_tokens": (usage_in or 0) + (usage_out or 0)}}))
        frames.append("data: [DONE]\n\n")
        return "".join(frames)
    frames.append(sse_frame("message_start", {"type": "message_start", "message": {
        "id": "msg_agg", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": 0}}}))
    frames.append(sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}))
    for piece in pieces:
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {"type": "message_delta",
                                              "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                              "usage": {"output_tokens": usage_out}}))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


class State:
    def __init__(self, brief, catalog, env_url):
        self.brief = brief
        self.catalog = catalog
        self.env_url = env_url
        self.client_version = "2023-06-01"
        self.served = 0  # client requests answered so far (manifest order)
        self.consec_fail = {}  # provider id -> consecutive refused calls
        self.skip_left = {}  # provider id -> requests it still sits out
        self.lock = threading.Lock()

    def route(self, req):
        """Serve one client request; returns (status, body_text, content_type)."""
        with self.lock:
            index = self.served
            self.served += 1
        manifest = list(self.brief.manifest.values())
        meta = manifest[index] if index < len(manifest) else {"requires": {}}
        candidates = self.brief.candidates(meta.get("requires", {}), req["json_mode"], req["api"])
        # Best first, with one refinement: a provider that refused twice
        # in a row is most likely spent or out (a scheduled refusal is
        # rarely twice in a row), so it sits out three requests before
        # one retry. That keeps dead providers from taxing every later
        # request a call, and bounds the quality a live one can lose.
        skipped = []
        for row in candidates:
            pid = row["provider"]
            if self.skip_left.get(pid, 0) > 0:
                self.skip_left[pid] -= 1
                skipped.append(row)
                continue
            answer = self._attempt(pid, row["model"], req)
            if answer is not None:
                return answer
        for row in skipped:  # nothing else answered: sitting out is over
            answer = self._attempt(row["provider"], row["model"], req)
            if answer is not None:
                return answer
        if req["api"] == "openai":
            return 429, json.dumps({"error": {"message": "every free provider refused",
                                               "type": "rate_limit_error", "param": None,
                                               "code": "rate_limit_exceeded"}}), "application/json"
        return 429, json.dumps({"type": "error", "error": {
            "type": "rate_limit_error", "message": "every free provider refused"}}), "application/json"

    def _attempt(self, pid, model, req):
        provider = self.brief.providers[pid]
        status, answer = call_provider(self, provider, model, req)
        if answer is None:
            self.consec_fail[pid] = self.consec_fail.get(pid, 0) + 1
            if self.consec_fail[pid] >= 2:
                self.skip_left[pid] = 3
            return None
        self.consec_fail[pid] = 0
        self.skip_left[pid] = 0
        text, usage_in, usage_out = answer
        if req["stream"]:
            return 200, client_sse(req["api"], model, text, usage_in, usage_out), "text/event-stream; charset=utf-8"
        return 200, json.dumps(client_json(req["api"], model, text, usage_in, usage_out)), "application/json"


class Handler(BaseHTTPRequestHandler):
    state = None

    def log_message(self, *args):
        pass

    def _send(self, status, text, content_type):
        body = text.encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/v1/models":
            self._send(404, json.dumps({"error": "not_found"}), "application/json")
            return
        ids = self.state.catalog.free_ids
        if self.headers.get("x-api-key") is not None and self.headers.get("authorization") is None:
            doc = {"data": [{"type": "model", "id": mid, "display_name": mid,
                             "created_at": "2026-01-01T00:00:00Z", "free": True} for mid in ids],
                   "has_more": False}
        else:
            doc = {"object": "list", "data": [
                {"id": mid, "object": "model", "created": CREATED, "owned_by": "aggregator",
                 "free": True} for mid in ids]}
        self._send(200, json.dumps(doc), "application/json")

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode())
        except Exception:
            body = {}
        if self.path == "/v1/chat/completions":
            req = parse_openai(body)
        elif self.path == "/v1/messages":
            version = self.headers.get("anthropic-version")
            if version:
                self.state.client_version = version
            req = parse_anthropic(body)
        else:
            self._send(404, json.dumps({"error": "not_found"}), "application/json")
            return
        status, text, content_type = self.state.route(req)
        self._send(status, text, content_type)


# ---------------------------------------------------------------- main

def load_brief():
    with open(os.environ["MAKEFUN_PROMPT_FILE"]) as fh:
        return Brief(fh.read())


def main():
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    if "--list" in sys.argv:
        if env_url and os.environ.get("MAKEFUN_PROMPT_FILE"):
            brief = load_brief()
            print(Catalog(brief, env_url).text)
        else:
            print('{"providers":[]}')
        return 0
    brief = load_brief()
    catalog = Catalog(brief, env_url)
    state = State(brief, catalog, env_url)
    Handler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    register = '{"protocol_version":1,"port":%d,"catalog":%s}' % (port, catalog.text)
    req = urllib.request.Request(env_url + "/register", data=register.encode(), method="POST")
    req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except Exception:
        pass
    import time
    while True:
        try:
            with urllib.request.urlopen(env_url + "/work", timeout=10) as resp:
                work = json.loads(resp.read().decode())
            if work.get("done"):
                break
        except Exception:
            pass
        time.sleep(0.25)
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

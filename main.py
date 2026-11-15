import asyncio
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import random

import httpx
from openai import AsyncOpenAI, OpenAIError
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse

app = FastAPI()

# One shared connection pool for every request. Building a new client per
# request costs CPU (SSL setup) and opens fresh connections every time,
# which is what made the proxy slow down under many parallel requests.
# No connection cap, so any number of requests can stream at once.
http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=None,
        max_keepalive_connections=100,
        keepalive_expiry=60,
    ),
    timeout=httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=None),
)

_clients: dict = {}

def get_client(base_url: str, key: str) -> AsyncOpenAI:
    cache_key = (base_url, key)
    client = _clients.get(cache_key)
    if client is None:
        if len(_clients) > 256:
            _clients.clear()
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=key,
            max_retries=0,
            http_client=http_client,
        )
        _clients[cache_key] = client
    return client

@app.on_event("shutdown")
async def close_http_client():
    await http_client.aclose()

@app.get("/")
async def root():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET"
    }
    with open("index.html", "r", encoding="utf-8") as html:
        return HTMLResponse(content=html.read(), headers=headers)

@app.get("/proxy")
async def redirect_to_root():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET"
    }
    return RedirectResponse(url="/", headers=headers)

@app.options("/proxy")
async def preflight_handler():
    headers = {
        "Access-Control-Allow-Origin": "https://janitorai.com",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type",
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Max-Age": "600"
    }
    return Response(headers=headers)

@app.post("/proxy")
async def callApi(
    request: Request,
    url: str,
    reasoning: str = "true",
    reasoning_visibility: str = "",
    Authorization: str = Header(None)
):
    def normalize_base_url(raw_url: str) -> str:
        trimmed = raw_url.rstrip("/")
        suffix = "/chat/completions"
        if trimmed.endswith(suffix):
            trimmed = trimmed[: -len(suffix)]
        return trimmed.rstrip("/")

    url = normalize_base_url(url)

    def extra_body():
        if reasoning == "true":
            return {"chat_template_kwargs": {"thinking": True}}
        if reasoning == "false":
            return {"chat_template_kwargs": {"thinking": False}}
        return {}

    TRANSIENT_STATUS = {429, 500, 502, 503, 504, 529}
    TRANSIENT_MARKERS = (
        "resourceexhausted",
        "overloaded",
        "temporarily overloaded",
        "too many requests",
        "rate limit",
        "rate_limit"
    )

    def is_transient(e):
        msg = str(getattr(e, "body", None) or e).lower()
        status = getattr(e, "status_code", None)

        if status in TRANSIENT_STATUS:
            return True

        if any(marker in msg for marker in TRANSIENT_MARKERS):
            return True

        for code in TRANSIENT_STATUS:
            if f'"code":{code}' in msg or f"'code': {code}" in msg:
                return True

        return False

    def get_retry_after(e, default_delay):
        response = getattr(e, "response", None)

        if response is None:
            return default_delay

        try:
            value = response.headers.get("retry-after")
        except Exception:
            return default_delay

        if not value:
            return default_delay

        try:
            return max(0.0, float(value))
        except ValueError:
            pass

        try:
            retry_date = parsedate_to_datetime(value)

            if retry_date.tzinfo is None:
                retry_date = retry_date.replace(tzinfo=timezone.utc)

            delay = (
                retry_date - datetime.now(timezone.utc)
            ).total_seconds()

            return max(0.0, delay)
        except (TypeError, ValueError, OverflowError):
            return default_delay

    async def openai_stream_caller(
        data,
        url,
        key,
        max_retries=3,
        base_delay=1.5
    ):
        last_err = None

        # Async client on the shared pool: awaiting it yields the event
        # loop, so other requests keep being served while this one waits.
        client = get_client(url, key)

        for attempt in range(max_retries):
            completion = None

            try:
                request_data = dict(data)
                request_data["stream"] = True
                request_data["top_p"] = 0.95

                existing_extra_body = request_data.pop("extra_body", {}) or {}
                existing_extra_body.update(extra_body())

                completion = await client.chat.completions.create(
                    **request_data,
                    extra_body=existing_extra_body,
                )

                first_chunk = await completion.__anext__()
                return first_chunk, completion

            except StopAsyncIteration:
                return None, completion

            except OpenAIError as e:
                last_err = e

                status = getattr(e, "status_code", None)
                response = getattr(e, "response", None)

                print("NVIDIA STATUS:", status)
                print("NVIDIA BODY:", getattr(e, "body", None))

                if response is not None:
                    try:
                        print(
                            "NVIDIA HEADERS:",
                            dict(response.headers)
                        )
                    except Exception:
                        pass

                if is_transient(e) and attempt < max_retries - 1:
                    default_delay = base_delay * (2 ** attempt)
                    retry_after = get_retry_after(
                        e,
                        default_delay
                    )

                    delay = retry_after

                    if retry_after == default_delay:
                        delay += random.uniform(0.0, 0.5)

                    print(
                        f"Retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries - 1})"
                    )

                    await asyncio.sleep(delay)
                    continue

                raise HTTPException(
                    status_code=status or 502,
                    detail=str(getattr(e, "body", e))
                )

            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=str(e)
                )

        status = getattr(last_err, "status_code", None)

        raise HTTPException(
            status_code=status or 502,
            detail=str(
                getattr(
                    last_err,
                    "body",
                    last_err
                )
            )
        )

    async def completion_generator(first_chunk, completion):
        is_in_reasnoning = False

        async def chain():
            if first_chunk is not None:
                yield first_chunk
            async for c in completion:
                yield c

        try:
            async for chunk in chain():
                if not chunk.choices:
                    continue

                if reasoning_visibility == "true":
                    if getattr(
                        chunk.choices[0].delta,
                        "reasoning_content",
                        None
                    ) is not None:

                        if not is_in_reasnoning:
                            is_in_reasnoning = True
                            yield (
                                'data: '
                                '{"choices":[{"delta":{"content":'
                                '" <think>"}}]}\n\n'
                            )

                        chunk.choices[0].delta.content = (
                            chunk.choices[0].delta.reasoning_content
                        )

                    else:
                        if is_in_reasnoning:
                            is_in_reasnoning = False
                            yield (
                                'data: '
                                '{"choices":[{"delta":{"content":'
                                '" </think>"}}]}\n\n'
                            )

                yield (
                    "data: "
                    + chunk.model_dump_json()
                    + "\n\n"
                )

        except OpenAIError as e:
            err_msg = (
                str(e.body)
                if getattr(e, "body", None)
                else str(e)
            )

            err_msg = (
                err_msg
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", " ")
                .replace("\r", " ")
            )

            yield (
                'data: {"choices":[{"delta":{"content":"'
                + "⚠️ Proxy error: "
                + err_msg
                + '"}}]}\n\n'
            )

        except Exception as e:
            err_msg = (
                str(e)
                .replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", " ")
                .replace("\r", " ")
            )

            yield (
                'data: {"choices":[{"delta":{"content":"'
                + "⚠️ Proxy error: "
                + err_msg
                + '"}}]}\n\n'
            )

        else:
            yield "data: [DONE]\n\n"
            return

        finally:
            # Release the upstream connection, also when the client
            # disconnects mid-stream.
            try:
                await completion.close()
            except Exception:
                pass

        yield "data: [DONE]\n\n"

    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid or empty JSON request body"
        )

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=400,
            detail="JSON request body must be an object"
        )

    if Authorization is None:
        headers = {
            "Access-Control-Allow-Origin": "https://janitorai.com",
            "Access-Control-Allow-Credentials": "true"
        }

        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers=headers
        )

    key = Authorization.replace("Bearer ", "")

    headers = {
        "Access-Control-Allow-Origin": "https://janitorai.com",
        "Access-Control-Allow-Credentials": "true",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }

    try:
        first_chunk, completion = await openai_stream_caller(
            data,
            url,
            key
        )
    except HTTPException as e:
        if e.headers is None:
            e.headers = {}

        e.headers.update(headers)

        raise e
    else:
        return StreamingResponse(
            completion_generator(
                first_chunk,
                completion
            ),
            media_type="text/event-stream",
            headers=headers
        )

@app.options("/proxy/blank")
async def blank_preflight_handler():
    return Response(
        headers={
            "Access-Control-Allow-Origin": "https://janitorai.com",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Authorization, Content-Type",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "600"
        }
    )

@app.post("/proxy/blank")
async def returnBlank(
    request: Request,
    text: str = "placeholder"
):
    def streaming_generator():
        safe_text = (
            str(text)
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", " ")
            .replace("\n", " ")
        )

        yield (
            'data: {"choices":[{"delta":{"content":"'
            + safe_text
            + '"}}]}\n\n'
        )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        content=streaming_generator(),
        media_type="text/event-stream",
        headers={
            "Access-Control-Allow-Origin": "https://janitorai.com",
            "Access-Control-Allow-Credentials": "true",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

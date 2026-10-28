# Deployment

The NiceGUI interface runs locally or in Docker. Hosted deployment requires
access controls and sufficient memory for PDF conversion.

## Local use

```bash
python -m pip install -e ".[app]"
python -m app.main
```

Open `http://127.0.0.1:8080`. The default bind address is loopback. Set `HOST`
explicitly when the app needs to listen on another interface.

## Docker

```bash
docker build -t polycompie .
docker run --rm -p 127.0.0.1:8080:8080 polycompie
```

The image installs the app dependencies and downloads Docling and EasyOCR
weights during the build. It listens on `0.0.0.0` inside the container; the
command above publishes the port only on the host's loopback interface.
Image size, build time, and memory use depend on dependency versions and input
PDFs. A first conversion still requires loading the models into memory.

## Hosted use

Configure the platform to build the supplied `Dockerfile`, expose the selected
`PORT`, and use `/healthz` as a liveness probe. Set `PORT` explicitly if the
platform expects a different port, such as 7860. A successful health check
confirms that the web server is running, not that conversion or inference works.

Before exposing the service to other users:

- Terminate HTTPS at the hosting platform or reverse proxy. Entered API keys
  reach the application server before being used with the provider.
- Add authentication or restrict access at the proxy. The application has no
  built-in login or per-user authorization.
- Set request-size, rate, and resource limits. The UI allows up to 10 PDFs of
  50 MB each, but those controls are not a substitute for server-side limits.
  The conversion semaphore does not limit all uploads or LLM requests.
- Keep credentials out of the image, source tree, and logs. Users provide their
  own OpenAI keys; the web form requires a key rather than offering the server's
  environment key to visitors.

Bring-your-own-key operation does not remove hosting costs or the need to
protect the service. Do not expose a local Ollama server directly to the public
internet.

## Session data

PDF bytes, API keys, outputs, and UI state are held in server memory for each
client session. PDF conversion uses temporary files. The app does not provide a
persistent results database; users should download their results. Closing a tab
is not a guarantee that an active worker stops immediately or that all server
memory is immediately erased.

Hosted extraction sends cleaned article text and intermediate extraction data
to the selected model provider. Local Ollama inference instead uses the
configured Ollama server.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` locally; `0.0.0.0` in Docker | Listen address |
| `PORT` | `8080` | Listen port |
| `POLYCOMPIE_MAX_CONCURRENT_CONVERSIONS` | `1` | Concurrent PDF conversions |
| `POLYCOMPIE_SKIP_PARSER_WARMUP` | `0` | Set to `1` to skip Docling startup warmup |

`GET /healthz` returns `{"status": "ok"}`. The optional `app/.env` file is loaded
for local app settings; it must not be committed or copied into a Docker image.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

from pyhocon import ConfigFactory
from sanic import Sanic, response
from sanic.log import logger

from deepspeech_server.engine import SpeechToTextEngine
from deepspeech_server.models import Response, Error

# Load app configs and initialize DeepSpeech model
conf = ConfigFactory.parse_file("application.conf")
engine = SpeechToTextEngine(
    model_path=Path(conf["deepspeech.model"]).absolute().as_posix(),
    scorer_path=Path(conf["deepspeech.scorer"]).absolute().as_posix(),
)

# Initialze Sanic and ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=conf["server.threadpool.count"])
app = Sanic("deepspeech_server")


@app.route("/", methods=["GET"])
async def healthcheck(request):
    return response.text("Welcome to DeepSpeech Server!")


@app.websocket("/api/v1/stt")
async def stt(request, ws):
    logger.debug(f"Received {request.method} request at {request.path}")
    try:
        audio = await ws.recv()

        inference_start = perf_counter()
        text = await app.loop.run_in_executor(executor, lambda: engine.run(audio))
        inference_end = perf_counter() - inference_start

        await ws.send(json.dumps(Response(text, inference_end).__dict__))
        logger.debug(f"Completed {request.method} request at {request.path} in {inference_end} seconds")
    except Exception as e:
        logger.debug(f"Failed to process {request.method} request at {request.path}. The exception is: {str(e)}.")
        await ws.send(json.dumps(Error("Something went wrong").__dict__))

    await ws.close()


if __name__ == "__main__":
    app.run(
        host=conf["server.http.host"],
        port=conf["server.http.port"],
        access_log=True,
        debug=True,
    )

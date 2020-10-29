# deepspeech-server

This directory contains a simple service that receives audio data from clients, and serves the results
of DeepSpeech inference over a websocket. The server code in this project is a modified version of
[this GitHub project](https://github.com/zelo/deepspeech-rest-api).

Because STT transcriptions can typically be considered "long running tasks", using websockets for client-server 
communication provides several benefits:
1. Avoids all sorts of timeouts at several points in the path - for example at the client, server, load balancer and/or
proxy, etc.
2. Avoids the need for the client to poll the server for result, as well as avoids the complexity that is typically 
induced by a polling-based architecture

## Usage

The client-server request-response process looks like the following:

1. Client opens websocket _W_ to server at the endpoint specified above
2. Client sends _binary_ audio data via _W_
3. Server responds with transcribed text via _W_ once transcription process is completed. The server's response is 
   in JSON format.
4. Server closes _W_

The time _t_ taken by the transcription process depends on several factors, such as the duration of the audio, how busy
the service is, etc. Under normal circumstances, _t_ is roughly the same as the duration of the provided audio.

Because this service uses websockets, it is currently not possible to interact with it using certain HTTP clients
which do not support websockets, like `curl`. The following example uses the
Python [`websocket-client`](https://pypi.org/project/websocket_client/) package.

```python
import websocket
    
ws = websocket.WebSocket()
ws.connect("ws://localhost:8080/api/v1/stt")

with open("audiofile.wav", mode='rb') as file:  # b is important -> binary
    audio = file.read()
    ws.send_binary(audio)
    result =  ws.recv()
    print(result) # Print text transcription received from server
```

Example output:
```
{"text": "experience proves this", "time": 2.4083645999999987}
```

## Deployment

### Kubernetes

The [helm](helm) directory contains an example Helm deployment, that configures an Nginx ingress to expose the 
DeepSpeech service. The websocket timeout on the ingress is set to 1 hour.

## Contributing

Bug reports and merge requests are welcome.

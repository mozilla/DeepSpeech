# IRC training notification bot

An IRC bot that notifies on training status and completed epochs.

## Requirements

* [Node.js](https://nodejs.org/)

## Preparation

For preparation of the bot, you need to run something around the following lines:

```bash
cd tools/trainerbot
npm install
```

## Running

Please use `--help` for getting the command line syntax.
Here is an example on how to run it:

```bash
./trainerbot.js some.irc.server.host 6665 \#teamchannel ../../.deepspeech_wer.out.log
```

Please refer to `README.website.md` regarding the expected logfile format.

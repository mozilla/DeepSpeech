Full project description and documentation on GitHub: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

## Generating TypeScript Type Definitions

You can generate the TypeScript type declaration file using `dts-gen`.
This requires a compiled/installed version of the DeepSpeech NodeJS client.

To generate a new `index.d.ts` type declaration file, run:

```sh
npm install -g dts-gen
dts-gen --module deepspeech --file index.d.ts
```

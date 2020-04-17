Full project description and documentation on GitHub: [https://github.com/mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech).

## Generating TypeScript Type Definitions

You can generate the TypeScript type declaration file using `dts-gen`.
This requires a compiled/installed version of the DeepSpeech NodeJS client.

Upon API change, it is required to generate a new `index.d.ts` type declaration
file, you have to run:

```sh
npm install -g dts-gen
dts-gen --module deepspeech --file index.d.ts
```

### Example usage

See `client.ts`

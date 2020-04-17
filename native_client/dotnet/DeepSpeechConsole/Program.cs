using DeepSpeechClient;
using DeepSpeechClient.Interfaces;
using DeepSpeechClient.Models;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace CSharpExamples
{
    class Program
    {
        /// <summary>
        /// Get the value of an argurment.
        /// </summary>
        /// <param name="args">Argument list.</param>
        /// <param name="option">Key of the argument.</param>
        /// <returns>Value of the argument.</returns>
        static string GetArgument(IEnumerable<string> args, string option)
        => args.SkipWhile(i => i != option).Skip(1).Take(1).FirstOrDefault();

        static string MetadataToString(CandidateTranscript transcript)
        {
            var nl = Environment.NewLine;
            string retval =
             Environment.NewLine + $"Recognized text: {string.Join("", transcript?.Tokens?.Select(x => x.Text))} {nl}"
             + $"Confidence: {transcript?.Confidence} {nl}"
             + $"Item count: {transcript?.Tokens?.Length} {nl}"
             + string.Join(nl, transcript?.Tokens?.Select(x => $"Timestep : {x.Timestep} TimeOffset: {x.StartTime} Char: {x.Text}"));
            return retval;
        }

        static void Main(string[] args)
        {
            string model = null;
            string scorer = null;
            string audio = null;
            bool extended = false;
            if (args.Length > 0)
            {
                model = GetArgument(args, "--model");
                scorer = GetArgument(args, "--scorer");
                audio = GetArgument(args, "--audio");
                extended = !string.IsNullOrWhiteSpace(GetArgument(args, "--extended"));
            }

            Stopwatch stopwatch = new Stopwatch();
            try
            {
                Console.WriteLine("Loading model...");
                stopwatch.Start();
                // sphinx-doc: csharp_ref_model_start
                using (IDeepSpeech sttClient = new DeepSpeech(model ?? "output_graph.pbmm"))
                {
                // sphinx-doc: csharp_ref_model_stop
                    stopwatch.Stop();

                    Console.WriteLine($"Model loaded - {stopwatch.Elapsed.Milliseconds} ms");
                    stopwatch.Reset();
                    if (scorer != null)
                    {
                        Console.WriteLine("Loading scorer...");
                        sttClient.EnableExternalScorer(scorer ?? "kenlm.scorer");
                    }

                    string audioFile = audio ?? "arctic_a0024.wav";
                    var waveBuffer = new WaveBuffer(File.ReadAllBytes(audioFile));
                    using (var waveInfo = new WaveFileReader(audioFile))
                    {
                        Console.WriteLine("Running inference....");

                        stopwatch.Start();

                        string speechResult;
                        // sphinx-doc: csharp_ref_inference_start
                        if (extended)
                        {
                            Metadata metaResult = sttClient.SpeechToTextWithMetadata(waveBuffer.ShortBuffer,
                                Convert.ToUInt32(waveBuffer.MaxSize / 2), 1);
                            speechResult = MetadataToString(metaResult.Transcripts[0]);
                        }
                        else
                        {
                            speechResult = sttClient.SpeechToText(waveBuffer.ShortBuffer,
                                Convert.ToUInt32(waveBuffer.MaxSize / 2));
                        }
                        // sphinx-doc: csharp_ref_inference_stop

                        stopwatch.Stop();

                        Console.WriteLine($"Audio duration: {waveInfo.TotalTime.ToString()}");
                        Console.WriteLine($"Inference took: {stopwatch.Elapsed.ToString()}");
                        Console.WriteLine((extended ? $"Extended result: " : "Recognized text: ") + speechResult);
                    }
                    waveBuffer.Clear();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
        }
    }
}

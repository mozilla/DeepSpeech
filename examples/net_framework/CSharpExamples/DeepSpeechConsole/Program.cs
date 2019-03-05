using DeepSpeechClient;
using DeepSpeechClient.Interfaces;
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

        static void Main(string[] args)
        {
            string model = null;
            string alphabet = null;
            string lm = null;
            string trie = null;
            string audio = null;
            if (args.Length > 0)
            {
                model = GetArgument(args, "--model");
                alphabet = GetArgument(args, "--alphabet");
                lm = GetArgument(args, "--lm");
                trie = GetArgument(args, "--trie");
                audio = GetArgument(args, "--audio");
            }

            const uint N_CEP = 26;
            const uint N_CONTEXT = 9;
            const uint BEAM_WIDTH = 200;
            const float LM_ALPHA = 0.75f;
            const float LM_BETA = 1.85f;

            Stopwatch stopwatch = new Stopwatch();

            using (IDeepSpeech sttClient = new DeepSpeech())
            {
                var result = 1;
                Console.WriteLine("Loading model...");
                stopwatch.Start();
                try
                { 
                    result = sttClient.CreateModel(
                        model ?? "output_graph.pbmm", 
                        N_CEP, N_CONTEXT,
                        alphabet ?? "alphabet.txt",
                        BEAM_WIDTH);
                }
                catch (IOException ex)
                {
                    Console.WriteLine("Error loading lm.");
                    Console.WriteLine(ex.Message);
                }

                stopwatch.Stop();
                if (result == 0)
                {
                    Console.WriteLine($"Model loaded - {stopwatch.Elapsed.Milliseconds} ms");
                    stopwatch.Reset();
                    if (lm != null)
                    {
                        Console.WriteLine("Loadin LM...");
                        try
                        {
                            result = sttClient.EnableDecoderWithLM(
                                alphabet ?? "alphabet.txt",
                                lm ?? "lm.binary",
                                trie ?? "trie",
                                LM_ALPHA, LM_BETA);
                        }
                        catch (IOException ex)
                        {
                            Console.WriteLine("Error loading lm.");
                            Console.WriteLine(ex.Message);
                        }

                    }

                    string audioFile = audio ?? "arctic_a0024.wav";
                    var waveBuffer = new WaveBuffer(File.ReadAllBytes(audioFile));
                    using (var waveInfo = new WaveFileReader(audioFile))
                    {
                        Console.WriteLine("Running inference....");

                        stopwatch.Start();

                        string speechResult = sttClient.SpeechToText(waveBuffer.ShortBuffer, Convert.ToUInt32(waveBuffer.MaxSize / 2), 16000);

                        stopwatch.Stop();

                        Console.WriteLine($"Audio duration: {waveInfo.TotalTime.ToString()}");
                        Console.WriteLine($"Inference took: {stopwatch.Elapsed.ToString()}");
                        Console.WriteLine($"Recognized text: {speechResult}");
                    }
                    waveBuffer.Clear();
                }
                else
                {
                    Console.WriteLine("Error loding the model.");
                }
            }
        }
    }
}

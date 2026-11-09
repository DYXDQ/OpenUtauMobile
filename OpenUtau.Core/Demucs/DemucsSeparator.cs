using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenUtau.Core.Util;
using NAudio.Wave;

namespace OpenUtau.Core.Demucs;

public sealed class DemucsSeparator : IDisposable
{
    private readonly InferenceSession session;
    private readonly DemucsConfig config;
    private readonly string inputName;
    private readonly string outputName;
    private readonly int channels;

    public int SampleRate => config.sample_rate;
    public IReadOnlyList<string> Sources => config.sources;

    public DemucsSeparator(string dependencyPath)
    {
        if (string.IsNullOrWhiteSpace(dependencyPath) || !Directory.Exists(dependencyPath))
        {
            throw new DirectoryNotFoundException($"Demucs dependency was not found: {dependencyPath}");
        }

        string configPath = Path.Combine(dependencyPath, "demucs.yaml");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException("demucs.yaml was not found in the dependency.", configPath);
        }

        config = Yaml.DefaultDeserializer.Deserialize<DemucsConfig>(File.ReadAllText(configPath))
            ?? throw new InvalidDataException("demucs.yaml is empty or invalid.");
        ValidateConfig(config);

        string modelPath = Path.Combine(dependencyPath, config.model);
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Demucs model was not found: {config.model}", modelPath);
        }

        session = Onnx.getInferenceSession(modelPath);
        inputName = string.IsNullOrWhiteSpace(config.input_name)
            ? session.InputMetadata.Keys.Single()
            : config.input_name;
        outputName = string.IsNullOrWhiteSpace(config.output_name)
            ? session.OutputMetadata.Keys.Single()
            : config.output_name;
        if (!session.InputMetadata.ContainsKey(inputName) || !session.OutputMetadata.ContainsKey(outputName))
        {
            throw new InvalidDataException("demucs.yaml refers to an unknown ONNX input or output.");
        }

        channels = GetChannelCount(session.InputMetadata[inputName].Dimensions);
    }

    public static DemucsSeparator FromInstalledDependency(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("A dependency name is required.", nameof(name));
        }
        return new DemucsSeparator(Path.Combine(PathManager.Inst.DependencyPath, name));
    }

    public IReadOnlyDictionary<string, float[]> Separate(
        float[] interleavedSamples,
        CancellationToken cancellationToken = default,
        IProgress<double>? progress = null)
    {
        if (interleavedSamples is null)
        {
            throw new ArgumentNullException(nameof(interleavedSamples));
        }

        if (interleavedSamples.Length % channels != 0)
        {
            throw new ArgumentException($"Audio must contain {channels} interleaved channels.", nameof(interleavedSamples));
        }

        int inputFrames = interleavedSamples.Length / channels;
        int segmentFrames = Math.Max(1, (int)Math.Round(config.segment_seconds * config.sample_rate));
        int hopFrames = Math.Max(1, (int)Math.Round(segmentFrames * (1 - config.overlap)));
        int segmentCount = Math.Max(1, (int)Math.Ceiling(Math.Max(1, inputFrames - segmentFrames) / (double)hopFrames) + 1);
        var outputs = config.sources.Select(_ => new float[inputFrames * channels]).ToArray();
        var weights = new double[inputFrames];

        for (int segmentIndex = 0, start = 0; segmentIndex < segmentCount; segmentIndex++, start += hopFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int frames = Math.Min(segmentFrames, Math.Max(0, inputFrames - start));
            float[] chunk = new float[segmentFrames * channels];
            Array.Copy(interleavedSamples, start * channels, chunk, 0, frames * channels);
            float[,,] separated = Infer(chunk, segmentFrames);

            for (int frame = 0; frame < frames; frame++)
            {
                int destinationFrame = start + frame;
                double weight = HannWeight(frame, segmentFrames);
                weights[destinationFrame] += weight;
                for (int source = 0; source < outputs.Length; source++)
                {
                    for (int channel = 0; channel < channels; channel++)
                    {
                        outputs[source][destinationFrame * channels + channel] +=
                            (float)(separated[source, channel, frame] * weight);
                    }
                }
            }

            progress?.Report((segmentIndex + 1) * 100.0 / segmentCount);
            if (start + segmentFrames >= inputFrames)
            {
                break;
            }
        }

        for (int frame = 0; frame < inputFrames; frame++)
        {
            double weight = Math.Max(weights[frame], 1e-8);
            for (int source = 0; source < outputs.Length; source++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    outputs[source][frame * channels + channel] =
                        (float)(outputs[source][frame * channels + channel] / weight);
                }
            }
        }

        return config.sources
            .Select((name, index) => new KeyValuePair<string, float[]>(name, outputs[index]))
            .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase);
    }

    public static void WriteWave(string path, float[] samples, int sampleRate, int channels)
    {
        if (samples.Length % channels != 0)
        {
            throw new ArgumentException("Audio samples do not align with the channel count.", nameof(samples));
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        using var writer = new WaveFileWriter(path, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels));
        writer.WriteSamples(samples, 0, samples.Length);
    }

    private float[,,] Infer(float[] interleavedSamples, int frames)
    {
        var input = new DenseTensor<float>(GetInputShape(frames));
        for (int frame = 0; frame < frames; frame++)
        {
            for (int channel = 0; channel < channels; channel++)
            {
                input[0, channel, frame] = interleavedSamples[frame * channels + channel];
            }
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> result = session.Run([
            NamedOnnxValue.CreateFromTensor(inputName, input)
        ]);
        Tensor<float> output = result.Single(item => item.Name == outputName).AsTensor<float>();
        return ReadOutput(output, frames);
    }

    private int[] GetInputShape(int frames)
    {
        int rank = session.InputMetadata[inputName].Dimensions.Count();
        return rank switch
        {
            3 => [1, channels, frames],
            4 => [1, 1, channels, frames],
            _ => throw new InvalidDataException("Demucs ONNX input must have rank 3 or 4.")
        };
    }

    private float[,,] ReadOutput(Tensor<float> output, int frames)
    {
        int[] dimensions = output.Dimensions.ToArray();
        var result = new float[config.sources.Count, channels, frames];
        if (dimensions.Length == 4 && dimensions[1] == config.sources.Count &&
            dimensions[2] == channels && dimensions[3] >= frames)
        {
            for (int source = 0; source < config.sources.Count; source++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    for (int frame = 0; frame < frames; frame++)
                    {
                        result[source, channel, frame] = output[0, source, channel, frame];
                    }
                }
            }
            return result;
        }

        throw new InvalidDataException("Unsupported Demucs ONNX output shape.");
    }

    private static int GetChannelCount(IReadOnlyList<int> dimensions)
    {
        return dimensions.Count switch
        {
            3 when dimensions[1] > 0 => dimensions[1],
            4 when dimensions[2] > 0 => dimensions[2],
            _ => 2
        };
    }

    private static double HannWeight(int index, int length)
    {
        return length <= 1 ? 1 : Math.Max(1e-3, 0.5 - 0.5 * Math.Cos(2 * Math.PI * index / (length - 1)));
    }

    private static void ValidateConfig(DemucsConfig value)
    {
        if (value.sample_rate <= 0 || value.segment_seconds <= 0 || value.overlap < 0 || value.overlap >= 1)
        {
            throw new InvalidDataException("demucs.yaml contains invalid sample_rate, segment_seconds, or overlap.");
        }
        if (value.sources.Count == 0 || value.sources.Any(string.IsNullOrWhiteSpace))
        {
            throw new InvalidDataException("demucs.yaml must define at least one source.");
        }
    }

    public void Dispose()
    {
        session.Dispose();
    }
}

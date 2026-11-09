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
        inputName = ResolveTensorName(session.InputMetadata.Keys, config.input_name, "input");
        outputName = ResolveTensorName(session.OutputMetadata.Keys, config.output_name, "output");

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

    /// <summary>
    /// 分离交错排列的立体声（或与模型一致的声道数）采样，按配置的源名称返回同长度交错采样。
    /// </summary>
    public IReadOnlyDictionary<string, float[]> Separate(
        float[] interleavedSamples,
        int audioChannels,
        CancellationToken cancellationToken = default,
        IProgress<double>? progress = null)
    {
        if (interleavedSamples is null)
        {
            throw new ArgumentNullException(nameof(interleavedSamples));
        }

        if (audioChannels <= 0 || interleavedSamples.Length % audioChannels != 0)
        {
            throw new ArgumentException($"Audio must contain {audioChannels} interleaved channels.", nameof(interleavedSamples));
        }

        if (audioChannels != channels)
        {
            throw new ArgumentException(
                $"Demucs expects {channels} channels but the audio has {audioChannels}.", nameof(audioChannels));
        }

        int inputFrames = interleavedSamples.Length / channels;
        if (inputFrames == 0)
        {
            return config.sources
                .Select(name => new KeyValuePair<string, float[]>(name, []))
                .ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.OrdinalIgnoreCase);
        }

        int segmentFrames = GetSegmentFrames();
        int hopFrames = Math.Max(1, (int)Math.Round(segmentFrames * (1 - config.overlap)));
        int segmentCount = Math.Max(1, (int)Math.Ceiling(Math.Max(0, inputFrames - segmentFrames) / (double)hopFrames) + 1);
        var outputs = config.sources.Select(_ => new float[inputFrames * channels]).ToArray();
        var weights = new double[inputFrames];
        float[] chunk = new float[segmentFrames * channels];

        for (int segmentIndex = 0; segmentIndex < segmentCount; segmentIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // 以片段中心对齐：首尾片段向两侧补静音，避免开头和结尾被淡入淡出压低。
            int start = segmentIndex * hopFrames;
            int frame = 0;
            for (; frame < segmentFrames && start + frame < inputFrames; frame++)
            {
                Array.Copy(interleavedSamples, (start + frame) * channels, chunk, frame * channels, channels);
            }
            Array.Clear(chunk, frame * channels, (segmentFrames - frame) * channels);
            float[,,] separated = Infer(chunk, segmentFrames);

            double[] gains = BuildWindow(segmentFrames, start, inputFrames);
            for (frame = 0; frame < segmentFrames && start + frame < inputFrames; frame++)
            {
                int destinationFrame = start + frame;
                double weight = gains[frame];
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
        int[] dimensions = session.InputMetadata[inputName].Dimensions;
        if (dimensions.Length == 4 && dimensions[2] > 0 && dimensions[2] != channels)
        {
            throw new InvalidDataException(
                $"Demucs ONNX input expects {dimensions[2]} channels but the audio has {channels}.");
        }

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

    private static string ResolveTensorName(IEnumerable<string> names, string configured, string kind)
    {
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return names.Contains(configured)
                ? configured
                : throw new InvalidDataException($"demucs.yaml refers to an unknown ONNX {kind}: {configured}");
        }
        string[] available = names.ToArray();
        return available.Length == 1
            ? available[0]
            : throw new InvalidDataException($"The Demucs ONNX model has {available.Length} {kind}s; set {kind}_name in demucs.yaml.");
    }

    /// <summary>片段长度取自模型固定形状，配置值仅在模型为动态长度时使用；过长的片段按配置值切分。</summary>
    private int GetSegmentFrames()
    {
        int configured = Math.Max(1, (int)Math.Round(config.segment_seconds * config.sample_rate));
        int[] dimensions = session.InputMetadata[inputName].Dimensions;
        int axis = dimensions.Length switch
        {
            3 => 2,
            4 => 3,
            _ => throw new InvalidDataException("Demucs ONNX input must have rank 3 or 4.")
        };
        return dimensions[axis] > 0 ? dimensions[axis] : configured;
    }

    /// <summary>
    /// 片段加权窗：中间为 1，两侧按汉宁窗渐入渐出；音频首尾处不衰减，边界帧且不从头尾
    /// 起算的片段也保持全权重，使任意重叠比例下每个采样点都有非零权重。
    /// </summary>
    private static double[] BuildWindow(int segmentFrames, int start, int inputFrames)
    {
        var gains = new double[segmentFrames];
        for (int i = 0; i < segmentFrames; i++)
        {
            gains[i] = Math.Min(FadeIn(i), FadeOut(segmentFrames - 1 - i));
        }
        if (start == 0)
        {
            gains[0] = 1;
        }
        if (start + segmentFrames >= inputFrames)
        {
            gains[segmentFrames - 1] = 1;
        }
        return gains;
    }

    private static double FadeIn(int index)
    {
        // 首个采样直接满权重，避免把音频第一个采样压到 0。
        return index <= 0 ? 1 : 0.5 - 0.5 * Math.Cos(Math.PI * index / 64.0);
    }

    private static double FadeOut(int remaining)
    {
        return remaining <= 0 ? 1 : 0.5 - 0.5 * Math.Cos(Math.PI * remaining / 64.0);
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

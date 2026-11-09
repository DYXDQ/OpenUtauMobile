using System.Collections.Generic;

namespace OpenUtau.Core.Demucs;

public sealed class DemucsConfig
{
    public string model { get; set; } = "model.onnx";
    public int sample_rate { get; set; } = 44100;
    public double segment_seconds { get; set; } = 5;
    public double overlap { get; set; } = 0.25;
    public string input_name { get; set; } = "";
    public string output_name { get; set; } = "";
    public List<string> sources { get; set; } = ["drums", "bass", "other", "vocals"];
}

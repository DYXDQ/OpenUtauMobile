using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using OpenUtau.Core.Demucs;
using OpenUtau.Core;
using OpenUtau.Core.Pipeline;
using OpenUtau.Core.Ustx;
using OpenUtauMobile.Helpers;
using OpenUtauMobile.Services.Dialogs;
using OpenUtauMobile.Storage;
using Serilog;

namespace OpenUtauMobile.ViewModels;

public partial class EditorViewModel
{
    private bool replacingAudio;
    private bool separatingAudio;

    private async Task SeparateSelectedAudioAsync()
    {
        if (separatingAudio || _disposed || SelectedParts is not [UWavePart source])
        {
            ToastService.Enqueue(L.S("Demucs.SelectAudio"));
            return;
        }

        separatingAudio = true;
        UProject project = DocManager.Inst.Project;
        try
        {
            string? dependency = Directory.Exists(PathManager.Inst.DependencyPath)
                ? Directory.EnumerateDirectories(PathManager.Inst.DependencyPath)
                    .FirstOrDefault(path => File.Exists(Path.Combine(path, "demucs.yaml")))
                : null;
            if (dependency == null)
            {
                throw new InvalidOperationException(L.S("Demucs.DependencyMissing"));
            }

            if (source.Samples == null)
            {
                source.Load(project);
                await source.Peaks;
            }
            if (source.Samples == null || source.channels <= 0 || source.sampleRate <= 0)
            {
                throw new InvalidDataException(L.S("Demucs.AudioUnreadable"));
            }

            (double _, double _, int channels, float[] samples) = source.GetTrimmedSamples(project);
            IReadOnlyDictionary<string, float[]>? stems = null;
            await LoadingPopupService.RunAsync(L.S("Demucs.Separating"), 0, async (loading, cancellationToken) =>
            {
                stems = await Task.Run(() =>
                {
                    using var separator = new DemucsSeparator(dependency);
                    return separator.Separate(samples, cancellationToken, new Progress<double>(value =>
                        loading.UpdateProgress(value, L.S("Demucs.Separating"))));
                }, cancellationToken);
                loading.UpdateProgress(100, L.S("Demucs.Completed"));
            });
            if (stems == null)
            {
                throw new InvalidOperationException(L.S("Demucs.NoOutput"));
            }

            string directory = Path.GetDirectoryName(source.FilePath) ?? PathManager.Inst.CachePath;
            string prefix = Path.Combine(directory,
                $"{Path.GetFileNameWithoutExtension(source.FilePath)}_demucs");
            List<UWavePart> parts = [];
            foreach ((string stem, float[] stemSamples) in stems)
            {
                string path = $"{prefix}_{stem}.wav";
                DemucsSeparator.WriteWave(path, stemSamples, 44100, channels);
                UWavePart part = new()
                {
                    FilePath = path,
                    position = source.position,
                    trackNo = project.tracks.Count + parts.Count
                };
                part.Load(project);
                parts.Add(part);
            }

            DocManager.Inst.StartUndoGroup("Editor.Action.SeparateAudio");
            try
            {
                foreach (UWavePart part in parts)
                {
                    DocManager.Inst.ExecuteCmd(new AddTrackCommand(project, new UTrack(project)
                    {
                        TrackNo = part.trackNo,
                        TrackName = $"{source.DisplayName} - {Path.GetFileNameWithoutExtension(part.FilePath).Split('_').Last()}"
                    }));
                    DocManager.Inst.ExecuteCmd(new AddPartCommand(project, part));
                }
            }
            finally
            {
                DocManager.Inst.EndUndoGroup();
            }
            ToastService.Enqueue(string.Format(L.S("Demucs.Success"), parts.Count));
        }
        catch (Exception exception)
        {
            Log.Error(exception, "Failed to separate audio with Demucs");
            if (!_disposed)
                ErrorDialogService.Show(new ErrorDialogViewModel(new ErrorMessageNotification(exception)));
        }
        finally
        {
            separatingAudio = false;
        }
    }

    /// <summary>选择并完整加载替代音频，验证后一次提交，取消或失败不修改原片段。</summary>
    private async Task ReplaceSelectedAudioAsync()
    {
        if (replacingAudio || _disposed || SelectedParts is not [UWavePart source]) return;
        replacingAudio = true;
        UProject project = DocManager.Inst.Project;
        DocRevision revision = DocManager.Inst.Revision;
        try
        {
            string file = await FilePicker.PickSingleFileAsync(L.S("Editor.Action.ReplaceAudio"),
                ["*.mp3", "*.wav", "*.flac", "*.aac", "*.m4a", "*.ogg", "*.aiff", "*.aif", "*.aifc"]);
            if (string.IsNullOrEmpty(file)) return;
            if (_disposed) return;
            EnsureAudioReplacementCurrent(project, source, revision);

            // 独立时间轴避免后台读取时与工程编辑共享可变的速度、拍号数据。
            UProject timing = new()
            {
                tempos = project.tempos.Select(tempo => new UTempo(tempo.position, tempo.bpm)).ToList(),
                timeSignatures = project.timeSignatures.Select(signature => new UTimeSignature(
                    signature.barPosition, signature.beatPerBar, signature.beatUnit)).ToList()
            };
            timing.timeAxis.BuildSegments(timing);
            UWavePart replacement = new()
            {
                FilePath = file,
                position = source.position,
                trackNo = source.trackNo
            };
            await LoadingPopupService.RunAsync(L.S("AudioReplacement.Loading"), _ => Task.Run(async () =>
            {
                replacement.Load(timing);
                if (await replacement.Peaks == null || replacement.Samples == null)
                    throw new InvalidDataException(string.Format(L.S("AudioReplacement.Unreadable"), file));
            }));

            if (_disposed) return;
            EnsureAudioReplacementCurrent(project, source, revision);
            PlaybackManager.Inst.StopPlayback();
            DocManager.Inst.StartUndoGroup("Editor.Action.ReplaceAudio");
            try
            {
                DocManager.Inst.ExecuteCmd(new ReplacePartCommand(project, source, replacement));
                ReplaceSelectedParts([replacement]);
            }
            finally
            {
                DocManager.Inst.EndUndoGroup();
            }
            ToastService.Enqueue(L.S("AudioReplacement.Success"));
        }
        catch (Exception exception)
        {
            Log.Error(exception, "Failed to replace audio part");
            if (!_disposed)
                ErrorDialogService.Show(new ErrorDialogViewModel(new ErrorMessageNotification(exception)));
        }
        finally
        {
            replacingAudio = false;
        }
    }

    /// <summary>文件选择和加载期间工程发生变化时，拒绝向过期片段提交替换。</summary>
    private static void EnsureAudioReplacementCurrent(UProject project, UWavePart source, DocRevision revision)
    {
        if (!ReferenceEquals(DocManager.Inst.Project, project) || !project.parts.Contains(source) ||
            DocManager.Inst.Revision != revision || DocManager.Inst.HasOpenUndoGroup)
            throw new InvalidOperationException(L.S("AudioReplacement.ProjectChanged"));
    }
}

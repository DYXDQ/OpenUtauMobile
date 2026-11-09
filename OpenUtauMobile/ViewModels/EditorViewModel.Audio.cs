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

    /// <summary>把选中的音频片段按 Demucs 分离为多条音轨；取消或失败不修改工程。</summary>
    private async Task SeparateSelectedAudioAsync()
    {
        if (separatingAudio)
        {
            ToastService.Enqueue(L.S("Demucs.AlreadyRunning"));
            return;
        }
        if (_disposed || SelectedParts is not [UWavePart source])
        {
            ToastService.Enqueue(L.S("Demucs.SelectAudio"));
            return;
        }

        separatingAudio = true;
        UProject project = DocManager.Inst.Project;
        DocRevision revision = DocManager.Inst.Revision;
        try
        {
            string dependency = DemucsDependencyLocator.Find()
                ?? throw new InvalidOperationException(L.S("Demucs.NeedDependency"));

            // 与音频替换一致：先完整加载（读取期间不阻塞 UI 线程），再取截取后的采样。
            if (source.Samples == null)
            {
                source.Load(project);
                if (source.Peaks != null)
                {
                    await source.Peaks;
                }
            }
            if (source.Samples == null || source.channels <= 0 || source.sampleRate <= 0)
            {
                throw new InvalidDataException(L.S("Demucs.AudioUnreadable"));
            }
            if (source.channels is < 1 or > 2)
            {
                throw new InvalidDataException(string.Format(L.S("Demucs.UnsupportedChannels"), source.channels));
            }

            // 独立时间轴：读取裁剪采样时不再共享工程里可变的曲速、拍号数据。
            UProject timing = CreateTimingProject(project, revision);
            int sampleRate = source.sampleRate;
            (double _, double _, int channels, float[] samples) = source.GetTrimmedSamples(timing);
            if (samples.Length == 0)
            {
                throw new InvalidDataException(L.S("Demucs.EmptyAudio"));
            }

            string directory = Path.GetDirectoryName(source.FilePath) ?? PathManager.Inst.CachePath;
            string prefix = Path.Combine(directory,
                $"{Path.GetFileNameWithoutExtension(source.FilePath)}_demucs");
            // 写盘与建片段都放进加载弹窗，避免进度停在 100% 时界面看起来卡住。
            List<(string Path, string Stem)> stemFiles = [];
            await LoadingPopupService.RunAsync(L.S("Demucs.Separating"), 0, async (loading, cancellationToken) =>
            {
                string[] names = [];
                IReadOnlyDictionary<string, float[]>? stems = await Task.Run(() =>
                {
                    using var separator = new DemucsSeparator(dependency);
                    names = separator.Sources.ToArray();
                    IProgress<double> progress = new Progress<double>(value =>
                        loading.UpdateProgress(value, L.S("Demucs.Separating")));
                    return separator.Separate(samples, channels, cancellationToken, progress);
                }, cancellationToken);
                if (stems == null || stems.Count == 0)
                {
                    throw new InvalidOperationException(L.S("Demucs.NoOutput"));
                }

                loading.UpdateProgress(100, L.S("Demucs.Writing"));
                stemFiles = await Task.Run(() =>
                {
                    // 按模型源的顺序写盘，保证音轨顺序稳定。
                    List<(string, string)> written = [];
                    foreach (string stem in names)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        string path = $"{prefix}_{stem}.wav";
                        DemucsSeparator.WriteWave(path, stems[stem], sampleRate, channels);
                        written.Add((path, stem));
                    }
                    return written;
                }, cancellationToken);
                loading.UpdateProgress(100, L.S("Demucs.Completed"));
            });
            if (stemFiles.Count == 0)
            {
                throw new InvalidOperationException(L.S("Demucs.NoOutput"));
            }

            if (_disposed)
            {
                return;
            }
            EnsureAudioReplacementCurrent(project, source, revision);
            PlaybackManager.Inst.StopPlayback();

            List<(UWavePart Part, string Stem)> parts = [];
            foreach ((string path, string stem) in stemFiles)
            {
                UWavePart part = new()
                {
                    FilePath = path,
                    position = source.position
                };
                part.Load(project);
                parts.Add((part, stem));
            }

            DocManager.Inst.StartUndoGroup("Editor.Action.SeparateAudio");
            try
            {
                foreach ((UWavePart part, string stem) in parts)
                {
                    part.trackNo = project.tracks.Count;
                    DocManager.Inst.ExecuteCmd(new AddTrackCommand(project, new UTrack(project)
                    {
                        TrackNo = part.trackNo,
                        TrackName = $"{Path.GetFileNameWithoutExtension(source.FilePath)} - {stem}"
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
        catch (OperationCanceledException)
        {
            Log.Information("Demucs separation cancelled");
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

    /// <summary>复制工程的曲速与拍号，供后台线程只读使用；工程已切换时拒绝继续。</summary>
    private static UProject CreateTimingProject(UProject project, DocRevision revision)
    {
        if (!ReferenceEquals(DocManager.Inst.Project, project) || DocManager.Inst.Revision != revision ||
            DocManager.Inst.HasOpenUndoGroup)
        {
            throw new InvalidOperationException(L.S("AudioReplacement.ProjectChanged"));
        }
        UProject timing = new()
        {
            tempos = project.tempos.Select(tempo => new UTempo(tempo.position, tempo.bpm)).ToList(),
            timeSignatures = project.timeSignatures.Select(signature => new UTimeSignature(
                signature.barPosition, signature.beatPerBar, signature.beatUnit)).ToList()
        };
        timing.timeAxis.BuildSegments(timing);
        return timing;
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

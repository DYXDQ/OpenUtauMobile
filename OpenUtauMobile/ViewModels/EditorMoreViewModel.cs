using System.Reactive;
using OpenUtau.Core;
using OpenUtau.Core.Demucs;
using OpenUtau.Core.Util;
using ReactiveUI;

namespace OpenUtauMobile.ViewModels;

public enum EditorMoreAction
{
    None, // 无操作
    ImportAudio, // 导入音频
    SeparateAudio, // Demucs 分离音频
    ImportTrack, // 导入轨道
    ExportAudio, // 导出音频
    SaveAsTemplate, // 保存为模板
    SaveAs, // 另存为
    Undo,
    Redo,
    ExpressionGraphs
}

public class EditorMoreViewModel : PopupViewModelBase
{
    public bool CanUndo => !DocManager.Inst.HasOpenUndoGroup && DocManager.Inst.GetUndoState(out _);
    public bool CanRedo => !DocManager.Inst.HasOpenUndoGroup && DocManager.Inst.GetRedoState(out _);

    /// <summary>已安装带 demucs.yaml 的依赖包时才允许分离音频，未安装时按钮置灰。</summary>
    public bool CanSeparateAudio => DemucsDependencyLocator.HasDependency();

    private bool _mergeNearbyPhrases = Preferences.Default.DiffSingerMergeNearbyPhrases;

    public bool MergeNearbyPhrases
    {
        get => _mergeNearbyPhrases;
        set
        {
            if (_mergeNearbyPhrases == value)
            {
                return;
            }
            this.RaiseAndSetIfChanged(ref _mergeNearbyPhrases, value);
            if (Preferences.Default.DiffSingerMergeNearbyPhrases == value)
            {
                return;
            }
            // 与桌面一致：先重新分句，再让预渲染读取新乐句的缓存键。
            Preferences.Default.DiffSingerMergeNearbyPhrases = value;
            Preferences.Save();
            DocManager.Inst.ExecuteCmd(new ValidateProjectNotification());
            DocManager.Inst.ExecuteCmd(new PreRenderNotification());
        }
    }

    public ReactiveCommand<EditorMoreAction, Unit> ConfirmCommand { get; }
    public ReactiveCommand<Unit, Unit> CancelCommand { get; }

    public EditorMoreViewModel()
    {
        ConfirmCommand = ReactiveCommand.Create<EditorMoreAction>(action => { RaiseClose(action); });

        CancelCommand = ReactiveCommand.Create(() => { RaiseClose(EditorMoreAction.None); });
    }
}

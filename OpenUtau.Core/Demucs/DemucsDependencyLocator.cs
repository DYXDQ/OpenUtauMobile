using System;
using System.IO;
using System.Linq;

namespace OpenUtau.Core.Demucs;

/// <summary>在依赖目录中定位已安装的 Demucs 依赖包（含 demucs.yaml 的目录）。</summary>
public static class DemucsDependencyLocator
{
    /// <summary>返回第一个包含 demucs.yaml 的依赖目录；未安装时返回 null。</summary>
    public static string? Find()
    {
        string dependencies = PathManager.Inst.DependencyPath;
        if (!Directory.Exists(dependencies))
        {
            return null;
        }
        try
        {
            return Directory.EnumerateDirectories(dependencies)
                .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault(path => File.Exists(Path.Combine(path, "demucs.yaml")));
        }
        catch (IOException)
        {
            return null;
        }
    }

    /// <summary>是否已安装可用作分离的 Demucs 依赖。</summary>
    public static bool HasDependency() => Find() != null;
}

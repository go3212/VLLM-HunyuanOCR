namespace HunyuanOCR.Client.Models;

/// <summary>
/// Server health status.
/// </summary>
public sealed class ServerStatus
{
    /// <summary>
    /// Whether the server is healthy.
    /// </summary>
    public bool Healthy { get; init; }

    /// <summary>
    /// Whether the model is loaded.
    /// </summary>
    public bool ModelLoaded { get; init; }

    /// <summary>
    /// Error message if the server is unhealthy.
    /// </summary>
    public string? Error { get; init; }
}


namespace HunyuanOCR.Client.Models;

/// <summary>
/// Configuration for HunyuanOCR client.
/// </summary>
public sealed class HunyuanOCRConfig
{
    /// <summary>
    /// The server URL. Default: http://localhost:8000
    /// </summary>
    public string ServerUrl { get; init; } = "http://localhost:8000";

    /// <summary>
    /// The model name. Default: tencent/HunyuanOCR
    /// </summary>
    public string Model { get; init; } = "tencent/HunyuanOCR";

    /// <summary>
    /// API key for authentication. Default: EMPTY
    /// </summary>
    public string ApiKey { get; init; } = "EMPTY";

    /// <summary>
    /// Maximum tokens for generation. Default: 16384
    /// </summary>
    public int MaxTokens { get; init; } = 16384;

    /// <summary>
    /// Temperature for generation. Default: 0.0
    /// </summary>
    public double Temperature { get; init; } = 0.0;

    /// <summary>
    /// Connection timeout in seconds. Default: 10.0
    /// </summary>
    public double ConnectTimeout { get; init; } = 10.0;

    /// <summary>
    /// Read timeout in seconds. Default: 120.0
    /// </summary>
    public double ReadTimeout { get; init; } = 120.0;

    /// <summary>
    /// Maximum concurrent connections. Default: 10
    /// </summary>
    public int MaxConnections { get; init; } = 10;

    /// <summary>
    /// Maximum concurrent workers for batch operations. Default: 4
    /// </summary>
    public int MaxWorkers { get; init; } = 4;

    /// <summary>
    /// Health check interval in seconds. Default: 2.0
    /// </summary>
    public double HealthCheckInterval { get; init; } = 2.0;

    /// <summary>
    /// Health check timeout in seconds. Default: 300.0 (5 minutes for model loading)
    /// </summary>
    public double HealthCheckTimeout { get; init; } = 300.0;
}


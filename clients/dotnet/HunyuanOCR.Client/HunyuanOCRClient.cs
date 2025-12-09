using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using HunyuanOCR.Client.Internal;
using HunyuanOCR.Client.Models;

namespace HunyuanOCR.Client;

/// <summary>
/// Async client for HunyuanOCR server.
/// </summary>
/// <example>
/// <code>
/// await using var client = new HunyuanOCRClient();
/// var result = await client.OcrImageAsync("document.png");
/// Console.WriteLine(result.Text);
/// </code>
/// </example>
public sealed class HunyuanOCRClient : IAsyncDisposable
{
    private readonly HunyuanOCRConfig _config;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private bool _disposed;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// Creates a new HunyuanOCR client with default configuration.
    /// </summary>
    public HunyuanOCRClient() : this(new HunyuanOCRConfig())
    {
    }

    /// <summary>
    /// Creates a new HunyuanOCR client with the specified configuration.
    /// </summary>
    public HunyuanOCRClient(HunyuanOCRConfig config) : this(config, null)
    {
    }

    /// <summary>
    /// Creates a new HunyuanOCR client with the specified configuration and HTTP client.
    /// </summary>
    /// <param name="config">Client configuration.</param>
    /// <param name="httpClient">Optional HTTP client. If null, a new one will be created.</param>
    public HunyuanOCRClient(HunyuanOCRConfig config, HttpClient? httpClient)
    {
        _config = config ?? throw new ArgumentNullException(nameof(config));
        
        if (httpClient is not null)
        {
            _httpClient = httpClient;
            _ownsHttpClient = false;
        }
        else
        {
            var handler = new SocketsHttpHandler
            {
                MaxConnectionsPerServer = config.MaxConnections,
                ConnectTimeout = TimeSpan.FromSeconds(config.ConnectTimeout),
                PooledConnectionLifetime = TimeSpan.FromMinutes(5)
            };
            
            _httpClient = new HttpClient(handler)
            {
                BaseAddress = new Uri(config.ServerUrl),
                Timeout = TimeSpan.FromSeconds(config.ReadTimeout)
            };
            _httpClient.DefaultRequestHeaders.Add("Accept", "application/json");
            _ownsHttpClient = true;
        }
    }

    /// <summary>
    /// Check if the server is healthy and model is loaded.
    /// </summary>
    public async Task<ServerStatus> HealthCheckAsync(CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            var response = await _httpClient.GetAsync("/health", cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                return new ServerStatus { Healthy = true, ModelLoaded = true, Error = null };
            }
            
            return new ServerStatus
            {
                Healthy = false,
                ModelLoaded = false,
                Error = $"Health check returned {(int)response.StatusCode}"
            };
        }
        catch (HttpRequestException ex)
        {
            return new ServerStatus
            {
                Healthy = false,
                ModelLoaded = false,
                Error = $"Connection error: {ex.Message}"
            };
        }
        catch (Exception ex)
        {
            return new ServerStatus
            {
                Healthy = false,
                ModelLoaded = false,
                Error = ex.Message
            };
        }
    }

    /// <summary>
    /// Wait for the server to be ready.
    /// </summary>
    /// <param name="timeout">Optional timeout override.</param>
    /// <param name="interval">Optional interval override.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>True if server became ready, false if timeout.</returns>
    public async Task<bool> WaitForReadyAsync(
        TimeSpan? timeout = null,
        TimeSpan? interval = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var effectiveTimeout = timeout ?? TimeSpan.FromSeconds(_config.HealthCheckTimeout);
        var effectiveInterval = interval ?? TimeSpan.FromSeconds(_config.HealthCheckInterval);
        
        var stopwatch = Stopwatch.StartNew();
        
        while (stopwatch.Elapsed < effectiveTimeout)
        {
            cancellationToken.ThrowIfCancellationRequested();
            
            var status = await HealthCheckAsync(cancellationToken);
            if (status.Healthy)
            {
                return true;
            }
            
            await Task.Delay(effectiveInterval, cancellationToken);
        }
        
        return false;
    }

    /// <summary>
    /// Perform OCR on an image file.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="prompt">OCR prompt type or custom prompt string.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult> OcrImageAsync(
        string imagePath,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxTokens = null,
        double? temperature = null,
        CancellationToken cancellationToken = default)
    {
        return await OcrImageAsync(
            imagePath,
            prompt.GetPromptText(),
            maxTokens,
            temperature,
            cancellationToken);
    }

    /// <summary>
    /// Perform OCR on an image file with a custom prompt.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult> OcrImageAsync(
        string imagePath,
        string promptText,
        int? maxTokens = null,
        double? temperature = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var imageBytes = await File.ReadAllBytesAsync(imagePath, cancellationToken);
        var mediaType = GetMediaType(imagePath);
        
        return await OcrImageBytesAsync(imageBytes, mediaType, promptText, maxTokens, temperature, cancellationToken);
    }

    /// <summary>
    /// Perform OCR on image bytes.
    /// </summary>
    /// <param name="imageBytes">The image bytes.</param>
    /// <param name="mediaType">The media type (e.g., "image/png").</param>
    /// <param name="prompt">OCR prompt type.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult> OcrImageBytesAsync(
        byte[] imageBytes,
        string mediaType,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxTokens = null,
        double? temperature = null,
        CancellationToken cancellationToken = default)
    {
        return await OcrImageBytesAsync(
            imageBytes,
            mediaType,
            prompt.GetPromptText(),
            maxTokens,
            temperature,
            cancellationToken);
    }

    /// <summary>
    /// Perform OCR on image bytes with a custom prompt.
    /// </summary>
    /// <param name="imageBytes">The image bytes.</param>
    /// <param name="mediaType">The media type (e.g., "image/png").</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult> OcrImageBytesAsync(
        byte[] imageBytes,
        string mediaType,
        string promptText,
        int? maxTokens = null,
        double? temperature = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var base64Image = Convert.ToBase64String(imageBytes);
        var dataUrl = $"data:{mediaType};base64,{base64Image}";
        
        var request = new ChatCompletionRequest
        {
            Model = _config.Model,
            Messages =
            [
                new ChatMessage { Role = "system", Content = "" },
                new ChatMessage
                {
                    Role = "user",
                    Content = new object[]
                    {
                        new ImageUrlContent { ImageUrl = new ImageUrl { Url = dataUrl } },
                        new TextContent { Text = promptText }
                    }
                }
            ],
            MaxTokens = maxTokens ?? _config.MaxTokens,
            Temperature = temperature ?? _config.Temperature
        };
        
        var response = await _httpClient.PostAsJsonAsync("/v1/chat/completions", request, JsonOptions, cancellationToken);
        response.EnsureSuccessStatusCode();
        
        var result = await response.Content.ReadFromJsonAsync<ChatCompletionResponse>(JsonOptions, cancellationToken)
            ?? throw new InvalidOperationException("Failed to deserialize response");
        
        var choice = result.Choices?.FirstOrDefault()
            ?? throw new InvalidOperationException("No choices in response");
        
        return new OCRResult
        {
            Text = choice.Message?.Content ?? string.Empty,
            Model = result.Model ?? _config.Model,
            PromptTokens = result.Usage?.PromptTokens ?? 0,
            CompletionTokens = result.Usage?.CompletionTokens ?? 0,
            TotalTokens = result.Usage?.TotalTokens ?? 0
        };
    }

    /// <summary>
    /// Perform OCR on multiple images concurrently.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="prompt">OCR prompt type.</param>
    /// <param name="maxConcurrency">Maximum concurrent requests.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult[]> OcrBatchAsync(
        IEnumerable<string> imagePaths,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxConcurrency = null,
        CancellationToken cancellationToken = default)
    {
        return await OcrBatchAsync(
            imagePaths,
            prompt.GetPromptText(),
            maxConcurrency,
            cancellationToken);
    }

    /// <summary>
    /// Perform OCR on multiple images concurrently with a custom prompt.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxConcurrency">Maximum concurrent requests.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task<OCRResult[]> OcrBatchAsync(
        IEnumerable<string> imagePaths,
        string promptText,
        int? maxConcurrency = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var paths = imagePaths.ToList();
        var concurrency = maxConcurrency ?? _config.MaxWorkers;
        var semaphore = new SemaphoreSlim(concurrency);
        
        var tasks = paths.Select(async path =>
        {
            await semaphore.WaitAsync(cancellationToken);
            try
            {
                return await OcrImageAsync(path, promptText, cancellationToken: cancellationToken);
            }
            finally
            {
                semaphore.Release();
            }
        });
        
        return await Task.WhenAll(tasks);
    }

    private static string GetMediaType(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".png" => "image/png",
            ".gif" => "image/gif",
            ".webp" => "image/webp",
            _ => "image/png"
        };
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        
        if (_ownsHttpClient)
        {
            _httpClient.Dispose();
        }
        
        await ValueTask.CompletedTask;
    }
}


using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using HunyuanOCR.Client.Internal;
using HunyuanOCR.Client.Models;

namespace HunyuanOCR.Client;

/// <summary>
/// Synchronous, thread-safe client for HunyuanOCR server.
/// </summary>
/// <example>
/// <code>
/// using var client = new HunyuanOCRClientSync();
/// var result = client.OcrImage("document.png");
/// Console.WriteLine(result.Text);
/// </code>
/// </example>
public sealed class HunyuanOCRClientSync : IDisposable
{
    private readonly HunyuanOCRConfig _config;
    private readonly HttpClient _httpClient;
    private readonly bool _ownsHttpClient;
    private readonly object _lock = new();
    private bool _disposed;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    /// <summary>
    /// Creates a new HunyuanOCR sync client with default configuration.
    /// </summary>
    public HunyuanOCRClientSync() : this(new HunyuanOCRConfig())
    {
    }

    /// <summary>
    /// Creates a new HunyuanOCR sync client with the specified configuration.
    /// </summary>
    public HunyuanOCRClientSync(HunyuanOCRConfig config) : this(config, null)
    {
    }

    /// <summary>
    /// Creates a new HunyuanOCR sync client with the specified configuration and HTTP client.
    /// </summary>
    /// <param name="config">Client configuration.</param>
    /// <param name="httpClient">Optional HTTP client. If null, a new one will be created.</param>
    public HunyuanOCRClientSync(HunyuanOCRConfig config, HttpClient? httpClient)
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
    public ServerStatus HealthCheck()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        try
        {
            var response = _httpClient.GetAsync("/health").GetAwaiter().GetResult();
            
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
    /// <returns>True if server became ready, false if timeout.</returns>
    public bool WaitForReady(TimeSpan? timeout = null, TimeSpan? interval = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var effectiveTimeout = timeout ?? TimeSpan.FromSeconds(_config.HealthCheckTimeout);
        var effectiveInterval = interval ?? TimeSpan.FromSeconds(_config.HealthCheckInterval);
        
        var stopwatch = Stopwatch.StartNew();
        
        while (stopwatch.Elapsed < effectiveTimeout)
        {
            var status = HealthCheck();
            if (status.Healthy)
            {
                return true;
            }
            
            Thread.Sleep(effectiveInterval);
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
    public OCRResult OcrImage(
        string imagePath,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxTokens = null,
        double? temperature = null)
    {
        return OcrImage(imagePath, prompt.GetPromptText(), maxTokens, temperature);
    }

    /// <summary>
    /// Perform OCR on an image file with a custom prompt.
    /// </summary>
    /// <param name="imagePath">Path to the image file.</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    public OCRResult OcrImage(
        string imagePath,
        string promptText,
        int? maxTokens = null,
        double? temperature = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var imageBytes = File.ReadAllBytes(imagePath);
        var mediaType = GetMediaType(imagePath);
        
        return OcrImageBytes(imageBytes, mediaType, promptText, maxTokens, temperature);
    }

    /// <summary>
    /// Perform OCR on image bytes.
    /// </summary>
    /// <param name="imageBytes">The image bytes.</param>
    /// <param name="mediaType">The media type (e.g., "image/png").</param>
    /// <param name="prompt">OCR prompt type.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    public OCRResult OcrImageBytes(
        byte[] imageBytes,
        string mediaType,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxTokens = null,
        double? temperature = null)
    {
        return OcrImageBytes(imageBytes, mediaType, prompt.GetPromptText(), maxTokens, temperature);
    }

    /// <summary>
    /// Perform OCR on image bytes with a custom prompt.
    /// </summary>
    /// <param name="imageBytes">The image bytes.</param>
    /// <param name="mediaType">The media type (e.g., "image/png").</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxTokens">Optional max tokens override.</param>
    /// <param name="temperature">Optional temperature override.</param>
    public OCRResult OcrImageBytes(
        byte[] imageBytes,
        string mediaType,
        string promptText,
        int? maxTokens = null,
        double? temperature = null)
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
        
        var response = _httpClient.PostAsJsonAsync("/v1/chat/completions", request, JsonOptions)
            .GetAwaiter().GetResult();
        response.EnsureSuccessStatusCode();
        
        var result = response.Content.ReadFromJsonAsync<ChatCompletionResponse>(JsonOptions)
            .GetAwaiter().GetResult()
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
    /// Perform OCR on multiple images using parallel processing.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="prompt">OCR prompt type.</param>
    /// <param name="maxWorkers">Maximum concurrent workers.</param>
    /// <param name="preserveOrder">Whether to preserve input order in results.</param>
    public List<OCRResult> OcrBatch(
        IEnumerable<string> imagePaths,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        int? maxWorkers = null,
        bool preserveOrder = true)
    {
        return OcrBatch(imagePaths, prompt.GetPromptText(), maxWorkers, preserveOrder);
    }

    /// <summary>
    /// Perform OCR on multiple images using parallel processing with a custom prompt.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="maxWorkers">Maximum concurrent workers.</param>
    /// <param name="preserveOrder">Whether to preserve input order in results.</param>
    public List<OCRResult> OcrBatch(
        IEnumerable<string> imagePaths,
        string promptText,
        int? maxWorkers = null,
        bool preserveOrder = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var paths = imagePaths.ToList();
        var workers = maxWorkers ?? _config.MaxWorkers;
        
        if (preserveOrder)
        {
            var results = new OCRResult[paths.Count];
            
            Parallel.For(0, paths.Count, new ParallelOptions { MaxDegreeOfParallelism = workers }, i =>
            {
                results[i] = OcrImage(paths[i], promptText);
            });
            
            return results.ToList();
        }
        else
        {
            var results = new System.Collections.Concurrent.ConcurrentBag<OCRResult>();
            
            Parallel.ForEach(paths, new ParallelOptions { MaxDegreeOfParallelism = workers }, path =>
            {
                results.Add(OcrImage(path, promptText));
            });
            
            return results.ToList();
        }
    }

    /// <summary>
    /// Perform batch OCR with callbacks for progress tracking.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="prompt">OCR prompt type.</param>
    /// <param name="callback">Callback invoked when an image is processed successfully.</param>
    /// <param name="errorCallback">Callback invoked when an error occurs.</param>
    /// <param name="maxWorkers">Maximum concurrent workers.</param>
    public List<OCRResult?> OcrBatchWithCallback(
        IEnumerable<string> imagePaths,
        OCRPromptType prompt = OCRPromptType.SpottingEn,
        Action<int, OCRResult>? callback = null,
        Action<int, Exception>? errorCallback = null,
        int? maxWorkers = null)
    {
        return OcrBatchWithCallback(imagePaths, prompt.GetPromptText(), callback, errorCallback, maxWorkers);
    }

    /// <summary>
    /// Perform batch OCR with callbacks for progress tracking and a custom prompt.
    /// </summary>
    /// <param name="imagePaths">Paths to the image files.</param>
    /// <param name="promptText">Custom prompt text.</param>
    /// <param name="callback">Callback invoked when an image is processed successfully.</param>
    /// <param name="errorCallback">Callback invoked when an error occurs.</param>
    /// <param name="maxWorkers">Maximum concurrent workers.</param>
    public List<OCRResult?> OcrBatchWithCallback(
        IEnumerable<string> imagePaths,
        string promptText,
        Action<int, OCRResult>? callback = null,
        Action<int, Exception>? errorCallback = null,
        int? maxWorkers = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var paths = imagePaths.ToList();
        var workers = maxWorkers ?? _config.MaxWorkers;
        var results = new OCRResult?[paths.Count];
        
        Parallel.For(0, paths.Count, new ParallelOptions { MaxDegreeOfParallelism = workers }, i =>
        {
            try
            {
                var result = OcrImage(paths[i], promptText);
                results[i] = result;
                callback?.Invoke(i, result);
            }
            catch (Exception ex)
            {
                results[i] = null;
                errorCallback?.Invoke(i, ex);
            }
        });
        
        return results.ToList();
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
    public void Dispose()
    {
        if (_disposed) return;
        
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            
            if (_ownsHttpClient)
            {
                _httpClient.Dispose();
            }
        }
    }
}


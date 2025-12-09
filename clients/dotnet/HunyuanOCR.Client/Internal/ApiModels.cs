using System.Text.Json.Serialization;

namespace HunyuanOCR.Client.Internal;

/// <summary>
/// Chat completion request payload.
/// </summary>
internal sealed class ChatCompletionRequest
{
    [JsonPropertyName("model")]
    public required string Model { get; init; }

    [JsonPropertyName("messages")]
    public required List<ChatMessage> Messages { get; init; }

    [JsonPropertyName("max_tokens")]
    public int MaxTokens { get; init; }

    [JsonPropertyName("temperature")]
    public double Temperature { get; init; }
}

/// <summary>
/// Chat message in a completion request.
/// </summary>
internal sealed class ChatMessage
{
    [JsonPropertyName("role")]
    public required string Role { get; init; }

    [JsonPropertyName("content")]
    public required object Content { get; init; }
}

/// <summary>
/// Image URL content part.
/// </summary>
internal sealed class ImageUrlContent
{
    [JsonPropertyName("type")]
    public string Type => "image_url";

    [JsonPropertyName("image_url")]
    public required ImageUrl ImageUrl { get; init; }
}

/// <summary>
/// Image URL object.
/// </summary>
internal sealed class ImageUrl
{
    [JsonPropertyName("url")]
    public required string Url { get; init; }
}

/// <summary>
/// Text content part.
/// </summary>
internal sealed class TextContent
{
    [JsonPropertyName("type")]
    public string Type => "text";

    [JsonPropertyName("text")]
    public required string Text { get; init; }
}

/// <summary>
/// Chat completion response from the API.
/// </summary>
internal sealed class ChatCompletionResponse
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("choices")]
    public List<ChatChoice>? Choices { get; init; }

    [JsonPropertyName("usage")]
    public UsageInfo? Usage { get; init; }
}

/// <summary>
/// A choice in the chat completion response.
/// </summary>
internal sealed class ChatChoice
{
    [JsonPropertyName("index")]
    public int Index { get; init; }

    [JsonPropertyName("message")]
    public ChatResponseMessage? Message { get; init; }

    [JsonPropertyName("finish_reason")]
    public string? FinishReason { get; init; }
}

/// <summary>
/// Message in a chat completion response.
/// </summary>
internal sealed class ChatResponseMessage
{
    [JsonPropertyName("role")]
    public string? Role { get; init; }

    [JsonPropertyName("content")]
    public string? Content { get; init; }
}

/// <summary>
/// Token usage information.
/// </summary>
internal sealed class UsageInfo
{
    [JsonPropertyName("prompt_tokens")]
    public int PromptTokens { get; init; }

    [JsonPropertyName("completion_tokens")]
    public int CompletionTokens { get; init; }

    [JsonPropertyName("total_tokens")]
    public int TotalTokens { get; init; }
}


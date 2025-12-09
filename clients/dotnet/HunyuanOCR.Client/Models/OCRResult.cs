namespace HunyuanOCR.Client.Models;

/// <summary>
/// Result from an OCR operation.
/// </summary>
public sealed class OCRResult
{
    /// <summary>
    /// The extracted text from the image.
    /// </summary>
    public required string Text { get; init; }

    /// <summary>
    /// The model used for OCR.
    /// </summary>
    public required string Model { get; init; }

    /// <summary>
    /// Number of tokens in the prompt.
    /// </summary>
    public int PromptTokens { get; init; }

    /// <summary>
    /// Number of tokens in the completion.
    /// </summary>
    public int CompletionTokens { get; init; }

    /// <summary>
    /// Total number of tokens used.
    /// </summary>
    public int TotalTokens { get; init; }
}


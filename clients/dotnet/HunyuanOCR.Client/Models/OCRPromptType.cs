namespace HunyuanOCR.Client.Models;

/// <summary>
/// Pre-defined OCR prompts for different tasks.
/// </summary>
public enum OCRPromptType
{
    /// <summary>
    /// English text spotting - detect and recognize text with coordinates.
    /// </summary>
    SpottingEn,

    /// <summary>
    /// Chinese text spotting - detect and recognize text with coordinates.
    /// </summary>
    SpottingZh,

    /// <summary>
    /// Formula recognition - output in LaTeX format.
    /// </summary>
    Formula,

    /// <summary>
    /// Table parsing - output in HTML format.
    /// </summary>
    Table,

    /// <summary>
    /// Chart parsing - Mermaid for flowcharts, Markdown for others.
    /// </summary>
    Chart,

    /// <summary>
    /// Document parsing - full document extraction in Markdown.
    /// </summary>
    Document,

    /// <summary>
    /// Subtitle extraction from images.
    /// </summary>
    Subtitles,

    /// <summary>
    /// Translation to English.
    /// </summary>
    TranslateEn
}

/// <summary>
/// Extension methods for OCRPromptType.
/// </summary>
public static class OCRPromptTypeExtensions
{
    private static readonly Dictionary<OCRPromptType, string> Prompts = new()
    {
        [OCRPromptType.SpottingEn] = "Detect and recognize text in the image, and output the text coordinates in a formatted manner.",
        [OCRPromptType.SpottingZh] = "检测并识别图片中的文字，将文本坐标格式化输出。",
        [OCRPromptType.Formula] = "Identify the formula in the image and represent it using LaTeX format.",
        [OCRPromptType.Table] = "Parse the table in the image into HTML.",
        [OCRPromptType.Chart] = "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.",
        [OCRPromptType.Document] = "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order.",
        [OCRPromptType.Subtitles] = "Extract the subtitles from the image.",
        [OCRPromptType.TranslateEn] = "First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format."
    };

    /// <summary>
    /// Gets the prompt text for the specified prompt type.
    /// </summary>
    public static string GetPromptText(this OCRPromptType promptType)
    {
        return Prompts[promptType];
    }
}


How to choose

| Your documents | Start with |
|---|---|
| Structured (headings, sections, tables): reports, records, manuals, contracts | Hybrid (structure + token limit) + headings |
| Plain text with paragraphs: articles, emails | Recursive |
| No structure at all: transcripts, OCR, logs | Fixed size with overlap, or semantic for long topic-drifting text |
| Code | By syntax (functions, classes) |
| Chunks that lose who/what they're about | Add a contextual header (from metadata if you have it, LLM otherwise), or late chunking |
| Answers spread over a section | Parent–child |

Then measure, which is the point of the exercise. In step 9 I used evidence coverage (does each piece of evidence survive inside one chunk?). It's a quick test you can run on any chunking before indexing anything. After that come the real retrieval metrics on the evaluation set.

For this corpus (hospital_rag): the documents are well structured, so the hybrid method is the right base, and the cutting itself barely matters above 256 tokens. The gains will come from what the chunks carry: the patient line (the "identity only in the header" trap), the table format, and parent–child for the split and buried questions.
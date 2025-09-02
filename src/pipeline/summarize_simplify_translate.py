from transformers import pipeline

# summarization
summarizer = pipeline("summarization", model="google/mt5-small")

# translation (English → Hindi)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")

def process(text: str):
    # summarize
    s = summarizer(text, max_length=60, min_length=10, do_sample=False)[0]["summary_text"]
    # translate
    hi = translator(s)[0]["translation_text"]
    return {"summary_en": s, "summary_hi": hi}

if __name__ == "__main__":
    sample = "The Ministry of Education released a new scheme to provide grants for rural schools."
    out = process(sample)
    print(out)

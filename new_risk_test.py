from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from functools import lru_cache

# =========================================================
# HUGGING FACE MODEL NAME
# =========================================================

model_name = "sohampal0011/risk-classifier"


@lru_cache(maxsize=1)
def get_classifier():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(text):
    classifier = get_classifier()
    result = classifier(text)

    label = result[0]["label"]

    # Return only label string
    return label

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    text = "i am safe but still afraid of the future"

    output = predict(text)

    print("\nTEXT:")
    print(text)

    print("\nPREDICTED RISK:")
    print(output)
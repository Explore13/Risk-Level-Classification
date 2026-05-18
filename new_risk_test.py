from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# HUGGING FACE MODEL NAME
# =========================================================

model_name = "sohampal0011/risk-classifier"

# =========================================================
# LOAD TOKENIZER
# =========================================================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

# =========================================================
# LOAD MODEL
# =========================================================

model = AutoModelForSequenceClassification.from_pretrained(
    model_name
)

# =========================================================
# CREATE PIPELINE
# =========================================================

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

print("✅ Model loaded successfully from Hugging Face!")

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(text):

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
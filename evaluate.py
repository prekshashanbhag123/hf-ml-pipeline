from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("This project is amazing!")
print(result)
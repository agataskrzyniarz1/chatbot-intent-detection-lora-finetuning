from transformers import pipeline

classifier = pipeline( 
  "text-classification",
  model="agataskrzyniarz/intent-detection-chatbot"
) 

classifier("Can you explain it to me?")

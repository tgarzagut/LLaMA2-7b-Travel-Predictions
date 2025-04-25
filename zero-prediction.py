#zero shot prediction for a singular person's work transportation mode

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# generation
input_text = """I will provide you with descriptors of a person then you will predict their travel mode for work from a pool of given options.
The only travel mode options are Bicycle or e-bike, Bus (public transit), Carpool, Commuter rail (Sounder, Amtrak), Drive alone, Motorcycle, and Walk.
Age: 25-34 years
Gender: Male
Race: White non-Hispanic
Employment status: Employed full time (35+ hours/week paid)
Workplace: Usually the same location (outside home)
Job count: 1 job
Commute frequency: 4 days a week
Commute Duration: Between 1 and 2 years
Telecommute frequency: Never
Hours worked: 31-40 hours
Education level: Bachelor degree
Adult student: No not a student
License: Yes has an intermediate or unrestricted license
Question: What is the main mode of transportation for work for this person?
Predicted Answer:"""

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("\nExpected answer: Bicycle or e-bike")

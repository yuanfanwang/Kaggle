 
text = """
Background 
The Third Wave experiment took place at Cubberley High School in Palo Alto, California during the first week of April 1967. History teacher Ron Jones, finding himself unable to explain to his students how people throughout history followed the crowd even when terrible things were happening, decided to demonstrate it to his students through an experiment. Jones announced that he was starting a movement aimed to eliminate democracy. Jones named the movement “The Third Wave” as a symbol of strength, referring to the mythical belief that the third in a series of waves is the strongest. One of the central points of this movement was that democracy’s main weakness is that it favors the individual over the whole community. Jones emphasized this main point of the movement when he created this catchy motto: “Strength through discipline, strength through community, strength through action, strength through pride.” 
The Experiment 
Jones started the first day of the experiment emphasizing simple things like proper seating, and drilled the students extensively until they got it right. He then proceeded to enforce strict classroom discipline by emerging as an authoritarian figure. This resulted in dramatic improvements to the efficiency, or orderliness, of the class.  The first day’s session ended with only a few rules. Jones intended it to be a one-day experiment. Students had to be sitting at attention before the second bell, had to stand up to ask or answer questions and had to do it in three words or fewer, and were required to preface each remark with “Mr. Jones.” As the week went on, Jones’ class transformed into a group with a supreme sense of discipline and community. Jones made up a salute resembling that of the Nazi regime and ordered class members to salute each other even outside the class. They all obeyed this command. 
After only three days, the experiment took on a life of its own, with students from all over the school joining in. The class expanded from initial 30 students to 43 attendees. All of the students showed drastic improvement in their academic skills and tremendous motivation. All of the students were issued a member card and each of them received a special assignment, like designing a Third Wave Banner, stopping non-members from entering the class, or other tasks to bring honor to the movement. Jones instructed the students on how to initiate new members, and by the end of the day the movement had over 200 participants. Jones was surprised that some of the students started reporting to him when other members of the movement failed to abide by the rules. 
By the fourth day of the experiment, the students became increasingly involved in the project and their discipline and loyalty to the project was so outstanding that Jones felt it was slipping out of control. He decided to terminate the movement, so he lied to students by announcing that the Third Wave was a part of a nationwide movement and that on the next day a presidential candidate of the movement would publicly announce its existence on television. Jones ordered students to attend a noon rally on Friday to witness the announcement. 
At the end of the week, instead of a televised address of their leader, the students were presented with a blank channel. After a few minutes of waiting, Jones announced that they had been a part of an experiment to demonstrate how people willingly create a sense of superiority over others, and how this can lead people to justify doing horrible things in the name of the state’s honor.
"""

# Install the Transformers library

# Import necessary modules
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the input text and the summary length
text = "This is a piece of text that you want to summarize."
max_length = 20

# Preprocess the text and encode it as input for the model
input_text = "summarize: " + text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate a summary
summary = model.generate(input_ids, max_length=max_length)

# Decode the summary
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)
print(summary_text)

import re

def simple_chatbot(user_input):
    # Convert user input to lowercase for case-insensitivity
    user_input = user_input.lower()

    # Define rules and responses
    rules = {
        'hello': 'Hi there! How can I help you?',
        'how are you': 'I am just a computer program, but I am doing well. How about you?',
        'goodbye': 'Goodbye! Have a great day!',
        'name': 'I am a chatbot. You can call me ChatGPT.',
        'default': 'I'm sorry, I don't understand that. Can you please rephrase or ask something else?'
    }

    # Check user input against rules
    for pattern, response in rules.items():
        if re.search(pattern, user_input):
            return response

    # Default response if no match is found
    return rules['default']

# Simple loop to run the chatbot
while True:
    # Get user input
    user_input = input("You: ")

    # Check for exit condition
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    # Get and print the chatbot's response
    response = simple_chatbot(user_input)
    print("Chatbot:", response)

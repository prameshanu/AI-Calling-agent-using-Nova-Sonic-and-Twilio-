# AI-Calling-agent-using-Nova-Sonic-and-Twilio-

# This code possess following features:
# 1. Incorporated the RAG to retrieve information from knowledge base(on intel-cs offerings) using the knowledgeBaseTool.
# 2. Added a new function to load the 'start.wav' audio clip so that Bot can speak first after initial user audio input with 'start'.
# 3. Dynamic prompt basis the customer name and their actions (visited websited/tried connecting with you etc).
# 4. Retaining the conversation history in the prompt for the agent to refer to.
# 5. Intent identification based on user input to determine if they are interested, not interested, have questions, or off-topic.
# 6. Upgraded knowledgeBaseTool to handle query and return results in a structured format.
# 7. Testing the prompts with direction not explicitly providing the conversation flow, but rather using the customer name and their actions to generate a dynamic conversation flow. 

# This code possess following features:
# 1. Incorporated the RAG to retrieve information from knowledge base(on intel-cs offerings) using the knowledgeBaseTool.
# 2. Added a new function to load the 'start.wav' audio clip so that Bot can speak first after initial user audio input with 'start'.
# 3. Dynamic prompt basis the customer name and their actions (visited websited/tried connecting with you etc).
# 4. Retaining the conversation history in the prompt for the agent to refer to.
# 5. Intent identification based on user input to determine if they are interested, not interested, have questions, or off-topic.
# 6. Upgraded knowledgeBaseTool to handle query and return results in a structured format.
# 7. Testing the prompts with direction not explicitly providing the conversation flow, but rather using the customer name and their actions to generate a dynamic conversation flow. 


import os
import asyncio
import base64
import json
import uuid
import warnings
import pyaudio
import pytz
import random
import hashlib
import datetime
import time
import inspect
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
# from prompt_file import prompt_intelcs
import webrtcvad
from collections import deque
import wave
from integration import bedrock_knowledge_bases as kb, br_agent_intel_cs_calling_1 as intro, conversation_flow as intro_flow, demo_time as demo_time
import re


# === Example Usage ===
default_details = """# Complete Conversation Flow for Nova Sonic AI Calling Agent (Mark)

**Customer Details:**
- Name: Mr. Michael
- Action: visited the website
- Time: couple of days ago
- Current Time: Good afternoon """

input_text=os.environ.get("input_text", default_details)

# Parse input
customer_info = intro_flow.parse_input(input_text)

print(f"Customer Info: {customer_info}")
# print(f"Customer Info: {customer_info(name)}")
# Generate dynamic output
conversation_flow = intro_flow.generate_dynamic_conversation_flow(customer_info, input_text)
# print(conversation_flow)
prompt_intelcs = f"""
{conversation_flow}
"""
print(f"Prompt Intel CS: {prompt_intelcs}")

# Suppress warnings
warnings.filterwarnings("ignore")

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of frames per buffer
# Debug mode flag
DEBUG = True

# customer_name=os.environ.get(customer_name, "Mr. John Smith")
# website_visit=os.environ.get(website_visit, True)
# tried_contact=os.environ.get(tried_contact, False)



def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        functionName = inspect.stack()[1].function
        if  functionName == 'time_it' or functionName == 'time_it_async':
            functionName = inspect.stack()[2].function
        print('{:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())[:-3] + ' ' + functionName + ' ' + message)

def time_it(label, methodToRun):
    start_time = time.perf_counter()
    result = methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

async def time_it_async(label, methodToRun):
    start_time = time.perf_counter()
    result = await methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result


class BedrockStreamManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""
    
    # Event templates
    START_SESSION_EVENT = '''{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
                }
            }
        }
    }'''

    CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }'''

    AUDIO_EVENT_TEMPLATE = '''{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TEXT_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": true,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
        }
    }'''

    TEXT_INPUT_EVENT = '''{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TOOL_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }'''

    CONTENT_END_EVENT = '''{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }'''

    PROMPT_END_EVENT = '''{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }'''

    SESSION_END_EVENT = '''{
        "event": {
            "sessionEnd": {}
        }
    }'''
    
    def start_prompt(self):
        """Create a promptStart event"""
        get_default_tool_schema = json.dumps({
            "type": "object",
            "properties": {},
            "required": []
        })

        
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        # "voiceId": "matthew",
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "getDateAndTimeTool",
                                    "description": "get information about date and time for demo",
                                    "inputSchema": {
                                        "json": get_default_tool_schema
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "knowledgeBaseTool",
                                    "description": "Runs query against a knowledge base to retrieve information.",
                                    "inputSchema": {
                                        "json": "{\"$schema\":\"http://json-schema.org/draft-07/schema#\",\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\",\"description\":\"the query to search\"}},\"required\":[\"query\"]}"
                                    }
                                }
                            }
                            
                        ]
                    }
                }
            }
        }
        
        return json.dumps(prompt_start_event)
    
    def tool_result_event(self, content_name, content, role):
        """Create a tool result event"""

        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content
            
        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content_json_string
                }
            }
        }
        return json.dumps(tool_result_event)
   
    def __init__(self, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        
        # Replace RxPy subjects with asyncio queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.barge_in = False
        self.bedrock_client = None
        
        # Audio playback components
        self.audio_player = None
        
        # Text response components
        self.display_assistant_text = False
        self.role = None

        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        self.conversation_context = {
            "history" : [],
            "user_intent" : None,
            "last_user_input" : None
        }


    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    # New function added to load the 'start.wav' audio clip
    def load_start_audio(self):
        """Load the 'start.wav' audio clip."""
        try:
            with wave.open('start.wav', 'rb') as wav_file:
                if (wav_file.getnchannels() != CHANNELS or
                    wav_file.getsampwidth() != 2 or  # 16-bit PCM
                    wav_file.getframerate() != INPUT_SAMPLE_RATE):
                    raise ValueError("start.wav must be 16 kHz, 16-bit PCM, mono")
                audio_data = wav_file.readframes(wav_file.getnframes())
            return audio_data
        except FileNotFoundError:
            print("Error: start.wav not found in the project directory.")
            raise
        except ValueError as e:
            print(f"Error: {e}")
            raise


    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        if not self.bedrock_client:
            self._initialize_client()
        
        try:
            self.stream_response = await time_it_async("invoke_model_with_bidirectional_stream", lambda : self.bedrock_client.invoke_model_with_bidirectional_stream( InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)))
            self.is_active = True
            # values = { "conversation_flow_cleaned": conversation_flow_cleaned}
            # default_system_prompt = prompt_intelcs.format(**values)
            default_system_prompt = prompt_intelcs.replace('"', '\\"').replace('\n', '\\n')

            # print(f"Default system prompt: {default_system_prompt}")
            # Send initialization events
            prompt_event = self.start_prompt()
            text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM")
            text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt)
            text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
            
            init_events = [self.START_SESSION_EVENT, prompt_event, text_content_start, text_content, text_content_end]
            
            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events
                await asyncio.sleep(0.1)
            
            # New function added to send initial user audio input with "start"
            start_audio_content_name = str(uuid.uuid4())
            audio_content_start = f'''
            {{
                "event": {{
                    "contentStart": {{
                        "promptName": "{self.prompt_name}",
                        "contentName":      "{start_audio_content_name}",
                        "type": "AUDIO",
                        "interactive": true,
                        "role": "USER",
                        "audioInputConfiguration": {{
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64"
                        }}
                    }}
                }}
            }}
            '''
            await self.send_raw_event(audio_content_start)
            
            # Load and send the "start" audio clip
            start_audio = self.load_start_audio()
            blob = base64.b64encode(start_audio)
            audio_event = f'''
            {{
                "event": {{
                    "audioInput": {{
                        "promptName": "{self.prompt_name}",
                        "contentName": "{start_audio_content_name}",
                        "content": "{blob.decode('utf-8')}"
                    }}
                }}
            }}
            '''
            await self.send_raw_event(audio_event)
            
            # Send content end for initial user audio
            audio_content_end = f'''
            {{
                "event": {{
                    "contentEnd": {{
                        "promptName": "{self.prompt_name}",
                        "contentName": "{start_audio_content_name}"
                    }}
                }}
            }}
            '''
            await self.send_raw_event(audio_content_end)



            # To start the mic when bot completed greeting and first message then only Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start processing audio input
            asyncio.create_task(self._process_audio_input())
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            debug_print("Stream initialized successfully")
            return self
        except Exception as e:
            self.is_active = False
            print(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            debug_print("Stream not initialized or closed")
            return
       
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            # For debugging large events, you might want to log just the type
            if DEBUG:
                if len(event_json) > 200:
                    event_type = json.loads(event_json).get("event", {}).keys()
                    debug_print(f"Sent event type: {list(event_type)}")
                else:
                    debug_print(f"Sent event: {event_json}")
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def send_audio_content_start_event(self):
        """Send a content start event to the Bedrock stream."""
        content_start_event = self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_start_event)
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        while self.is_active:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    debug_print("No audio bytes received")
                    continue
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self.AUDIO_EVENT_TEMPLATE % (
                    self.prompt_name, 
                    self.audio_content_name, 
                    blob.decode('utf-8')
                )
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        self.audio_input_queue.put_nowait({
            'audio_bytes': audio_bytes,
            'prompt_name': self.prompt_name,
            'content_name': self.audio_content_name
        })
    
    async def send_audio_content_end_event(self):
        """Send a content end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_end_event)
        debug_print("Audio ended")
    
    async def send_tool_start_event(self, content_name):
        """Send a tool content start event to the Bedrock stream."""
        content_start_event = self.TOOL_CONTENT_START_EVENT % (self.prompt_name, content_name, self.toolUseId)
        debug_print(f"Sending tool start event: {content_start_event}")  
        await self.send_raw_event(content_start_event)

    async def send_tool_result_event(self, content_name, tool_result):
        """Send a tool content event to the Bedrock stream."""
        # Use the actual tool result from processToolUse
        tool_result_event = self.tool_result_event(content_name=content_name, content=tool_result, role="TOOL")
        debug_print(f"Sending tool result event: {tool_result_event}")
        await self.send_raw_event(tool_result_event)
    
    async def send_tool_content_end_event(self, content_name):
        """Send a tool content end event to the Bedrock stream."""
        tool_content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, content_name)
        debug_print(f"Sending tool content event: {tool_content_end_event}")
        await self.send_raw_event(tool_content_end_event)
    
    async def send_prompt_end_event(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        prompt_end_event = self.PROMPT_END_EVENT % (self.prompt_name)
        await self.send_raw_event(prompt_end_event)
        debug_print("Prompt ended")
        
    async def send_session_end_event(self):
        """Send a session end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return

        await self.send_raw_event(self.SESSION_END_EVENT)
        self.is_active = False
        debug_print("Session ended")
    
    def infer_intent(self, user_input):
        """Infer user intent based on input text."""
        user_input = user_input.lower()
        intents = {
            "interested": r"(demo|schedule|interested|show me|try it)",
            "not_interested": r"(not interested|no thanks|not now)",
            "question": r"(how|what|why|tell me|explain)",
            "provide_email": r"[\w\.-]+@[\w\.-]+\.\w+",
            "off_topic": r"(bye|something else|random)"
        }

        for intent, pattern in intents.items():
            if re.search(pattern, user_input):
                self.conversation_context["user_intent"] = intent
                return intent
        self.conversation_context["user_intent"] = "unknown"
        return "unknown"

    async def send_text_input(self, text):
        # Infer intent before sending
        intent = self.infer_intent(text)
        debug_print(f"Detected intent: {intent}")
        """Send user text input with conversation context."""
        # Append user input to history
        self.conversation_context["history"].append({"role": "user", "content": text})
        self.conversation_context["last_user_input"] = text

        # Truncate history to avoid overflow (e.g., keep last 5 exchanges)
        if len(self.conversation_context["history"]) > 10:
            self.conversation_context["history"] = self.conversation_context["history"][-10:]

        # Include context in the text input event
        context_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.conversation_context["history"]])
        text_input = f"Conversation context:\n{context_summary}\n\nUser input: {text}"
        
        text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "USER")
        text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, text_input)
        text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)

        await self.send_raw_event(text_content_start)
        await self.send_raw_event(text_content)
        await self.send_raw_event(text_content_end)


    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in json_data:
                                if 'contentStart' in json_data['event']:
                                    debug_print("Content start detected")
                                    content_start = json_data['event']['contentStart']
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                debug_print("Speculative content detected")
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            debug_print("Error parsing additionalModelFields")
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    role = json_data['event']['textOutput']['role']
                                    # Check if there is a barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        debug_print("Barge-in detected. Stopping audio output.")
                                        self.barge_in = True

                                    if (self.role == "ASSISTANT"):
                                        self.conversation_context["history"].append({"role" : "ASSISTANT", "content": text_content})
                                        if self.display_assistant_text:
                                            print(f"Assistant: {text_content}")
                                    elif (self.role == "USER"):
                                        print(f"User: {text_content}")

                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    await self.audio_output_queue.put(audio_bytes)
                                elif 'toolUse' in json_data['event']:
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                    debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")
                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    debug_print("Processing tool use and sending result")
                                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    toolContent = str(uuid.uuid4())
                                    await self.send_tool_start_event(toolContent)
                                    await self.send_tool_result_event(toolContent, toolResult)
                                    await self.send_tool_content_end_event(toolContent)
                                
                                elif 'completionEnd' in json_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    print("End of response sequence")
                            
                            # Put the response in the output queue for other components
                            await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                   # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False

    async def processToolUse(self, toolName, toolUseContent):
        """Return the tool result"""
        debug_print(f"Tool Use Content: {toolUseContent}")
        tool = toolName.lower()
        content, result = None, None
        try:
            if toolUseContent.get("content"):
                # Parse the JSON string in the content field
                query_json = json.loads(toolUseContent.get("content"))
                content = toolUseContent.get("content")  # Pass the JSON string directly to the agent
                print(f"Extracted query: {content}")

            if tool == "getdateandtimetool":
                # Set the timezone to Dubai (Asia/Dubai)
                # Use UTC time and localize it properly
                utc_now = datetime.datetime.utcnow()
                dubai_timezone = pytz.timezone("Asia/Dubai")
                dubai_date = pytz.utc.localize(utc_now).astimezone(dubai_timezone)

                # Get current date in PST timezone
                #pst_timezone = pytz.timezone("America/Los_Angeles")
                #pst_date = datetime.datetime.now(pst_timezone)
                
                return {
                    "formattedTime": dubai_date.strftime("%I:%M %p"),
                    "date": dubai_date.strftime("%Y-%m-%d"),
                    "year": dubai_date.year,
                    "month": dubai_date.month,
                    "day": dubai_date.day,
                    "dayOfWeek": dubai_date.strftime("%A").upper(),
                    "timezone": "Gulf Standard Time"
                }
            
            if tool == "knowledgebasetool":
                result = kb.retrieve_kb(content)
                if result and isinstance(result, list) and len(result) > 0:
                # Summarize the first result in a conversational tone
                    summary = result[0].get("content", "No relevant info found.")
                    return {
                            "result": f"Here's what I found: {summary[:200]}... Want me to dive deeper?"
                    }
                return {"result": "I couldn't find anything on that. Can you clarify what you're looking for?"}

            return {"result": result if result else "No result found"}

            
            # if not result:
            #     result = "no result found"

            return {"result": result}
        except Exception as ex:
            print(ex)
            return {"result": "An error occurred while attempting to retrieve information related to the toolUse event."}
            
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
       
        self.is_active = False
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

        await self.send_audio_content_end_event()
        await self.send_prompt_end_event()
        await self.send_session_end_event()

        if self.stream_response:
            await self.stream_response.input_stream.close()
    
    async def send_barge_in_event(self):
        """Send events to interrupt the server."""
        if not self.is_active:
            debug_print("Stream not active, skipping barge-in")
            return
        try:
            debug_print("Sending audio content end event")
            await self.send_audio_content_end_event()
            self.audio_content_name = str(uuid.uuid4())
            debug_print("Sending audio content start event")
            await self.send_audio_content_start_event()
            debug_print("Barge-in events sent successfully")
        except Exception as e:
            debug_print(f"Error sending barge-in events: {e}")

class AudioStreamer:
    """Handles continuous microphone input and audio output using separate streams."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.loop = asyncio.get_event_loop()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(2)  # Moderate sensitivity
        self.audio_buffer = deque(maxlen=100)  # Buffer for input audio

        self.vad.set_mode(1)  # Lower sensitivity for less false positives
        self.speech_detected_count = 0
        self.speech_debounce_threshold = 3
        # Initialize PyAudio
        debug_print("AudioStreamer Initializing PyAudio...")
        self.p = time_it("AudioStreamerInitPyAudio", pyaudio.PyAudio)
        debug_print("AudioStreamer PyAudio initialized")
        self.chunk_size = 320  # 20ms at 16 kHz for VAD (320 samples)
        # Initialize separate streams for input and output
        # Input stream with callback for microphone
        debug_print("Opening input audio stream...")
        self.input_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.input_callback
        ))
        # debug_print("input audio stream opened")

        # Output stream for direct writing (no callback)
        debug_print("Opening output audio stream...")
        self.output_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        ))

        debug_print("output audio stream opened")

    # def input_callback(self, in_data, frame_count, time_info, status):
    #     """Callback function that schedules audio processing in the asyncio event loop"""
    #     if self.is_streaming and in_data:
    #         # Schedule the task in the event loop
    #         asyncio.run_coroutine_threadsafe(
    #             self.process_input_audio(in_data), 
    #             self.loop
    #         )
    #     return (None, pyaudio.paContinue)

    def input_callback(self, in_data, frame_count, time_info, status):
        if self.is_streaming and in_data:
            try:
                debug_print(f"Received input audio: {len(in_data)} bytes")
                is_speech = self.vad.is_speech(in_data, INPUT_SAMPLE_RATE)
                if is_speech:
                    self.speech_detected_count +=1
                    if self.speech_detected_count >= self.speech_debounce_threshold and not self.stream_manager.barge_in:
                        debug_print("Speech detected, triggering barge-in")
                        self.stream_manager.barge_in = True
                        asyncio.run_coroutine_threadsafe(
                            self.stream_manager.send_barge_in_event(), self.loop
                        )
                        asyncio.run_coroutine_threadsafe(
                            self.reset_barge_in_timeout(), self.loop
                        )
                else:
                    self.speech_detected_count = 0
                    self.audio_buffer.append(in_data)
                    asyncio.run_coroutine_threadsafe(
                        self.process_input_audio(in_data), self.loop
                    )
            except Exception as e:
                debug_print(f"Input callback error: {e}")
                asyncio.run_coroutine_threadsafe(
                self.process_input_audio(in_data), self.loop
            )
        return (None, pyaudio.paContinue)

    async def reset_barge_in_timeout(self):
        await asyncio.sleep(1.0)
        if self.stream_manager.barge_in:
            self.stream_manager.barge_in = False
            debug_print("Barge-in timeout, resetting flag")


    async def process_input_audio(self, audio_data):
        """Process a single audio chunk directly"""
        try:
            # Send audio to Bedrock immediately
            self.stream_manager.add_audio_chunk(audio_data)
        except Exception as e:
            if self.is_streaming:
                print(f"Error processing input audio: {e}")
    
    async def play_output_audio(self):
        """Play audio responses from Nova Sonic"""
        while self.is_streaming:
            try:
                # Check for barge-in flag
                if self.stream_manager.barge_in:
                    # Clear the audio queue
                    while not self.stream_manager.audio_output_queue.empty():
                        try:
                            self.stream_manager.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.stream_manager.barge_in = False
                    # Small sleep after clearing
                    await asyncio.sleep(0.05)
                    continue
                
                # Get audio data from the stream manager's queue
                audio_data = await asyncio.wait_for(
                    self.stream_manager.audio_output_queue.get(),
                    timeout=0.1
                )
                
                if audio_data and self.is_streaming:
                    # Write directly to the output stream in smaller chunks
                    chunk_size = CHUNK_SIZE  # Use the same chunk size as the stream
                    
                    # Write the audio data in chunks to avoid blocking too long
                    for i in range(0, len(audio_data), chunk_size):
                        if not self.is_streaming:
                            break
                        
                        end = min(i + chunk_size, len(audio_data))
                        chunk = audio_data[i:end]
                        
                        # Create a new function that captures the chunk by value
                        def write_chunk(data):
                            return self.output_stream.write(data)
                        
                        # Pass the chunk to the function
                        await asyncio.get_event_loop().run_in_executor(None, write_chunk, chunk)
                        
                        # Brief yield to allow other tasks to run
                        await asyncio.sleep(0.001)
                    
            except asyncio.TimeoutError:
                # No data available within timeout, just continue
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error playing output audio: {str(e)}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start_streaming(self):
        """Start streaming audio."""
        if self.is_streaming:
            return
        
        print("Starting audio streaming. Speak into your microphone...")
        print("Press Enter to stop streaming...")
        
        # Send audio content start event
        await time_it_async("send_audio_content_start_event", lambda : self.stream_manager.send_audio_content_start_event())
        
        self.is_streaming = True
        
        # Start the input stream if not already started
        if not self.input_stream.is_active():
            self.input_stream.start_stream()
        
        # Start processing tasks
        #self.input_task = asyncio.create_task(self.process_input_audio())
        self.output_task = asyncio.create_task(self.play_output_audio())
        
        # Wait for user to press Enter to stop
        await asyncio.get_event_loop().run_in_executor(None, input)
        
        # Once input() returns, stop streaming
        await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False

        # Cancel the tasks
        tasks = []
        if hasattr(self, 'input_task') and not self.input_task.done():
            tasks.append(self.input_task)
        if hasattr(self, 'output_task') and not self.output_task.done():
            tasks.append(self.output_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Stop and close the streams
        if self.input_stream:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
        if self.p:
            self.p.terminate()
        
        await self.stream_manager.close() 


async def main(debug=False):
    """Main function to run the application."""
    global DEBUG
    DEBUG = debug

    # Create stream manager
    stream_manager = BedrockStreamManager(model_id='amazon.nova-sonic-v1:0', region='us-east-1')

    # Create audio streamer
    audio_streamer = AudioStreamer(stream_manager)

    # Initialize the stream
    await time_it_async("initialize_stream", stream_manager.initialize_stream)

    try:
        # This will run until the user presses Enter
        await audio_streamer.start_streaming()
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        await audio_streamer.stop_streaming()
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Nova Sonic Python Streaming')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    # Set your AWS credentials here or use environment variables
    # os.environ['AWS_ACCESS_KEY_ID'] = "AWS_ACCESS_KEY_ID"
    # os.environ['AWS_SECRET_ACCESS_KEY'] = "AWS_SECRET_ACCESS_KEY"
    # os.environ['AWS_DEFAULT_REGION'] = "us-east-1"

    # Run the main function
    try:
        asyncio.run(main(debug=args.debug))
    except Exception as e:
        print(f"Application error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

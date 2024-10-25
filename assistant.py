from groq import Groq
from PIL import ImageGrab, Image
import google.generativeai as genai
import cv2
import pyperclip

# Initialize the Groq client with the API key
groq_client = Groq(api_key="api_key")  # Replace with your actual API key
genai.configure(api_key="api_key")
# Initialize the webcam
web_cam = cv2.VideoCapture(0)

sys_msg = (
    'You are a multi-model AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all the previous generated text in your response before '
    'adding the new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048,
}

safety_settings = [ 
    {
        'category': 'HARM_CATEGORY_HARASSMENT', 
        'threshold': 'BLOCK_NONE' 
    }, 
    { 
        'category': 'HARM_CATEGORY_HATE_SPEECH', 
        'threshold': 'BLOCK_NONE'
    },
    { 
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 
        'threshold': 'BLOCK_NONE'
    },
    { 
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 
        'threshold': 'BLOCK_NONE' 
    },
] 

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                                generation_config=generation_config,
                                safety_settings=safety_settings)

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead, take the user prompt input and try to extract all meaning from the photo '
        'relevant to the user prompt. Then generate as much objective data about the image for the AI '
        f'assistant who will respond to the user.\n USER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    return response.text

def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an A1 function calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam, or calling no functions is best for a voice assistant to respond '
        'to the user\'s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    # Define the conversation for the function call
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    # Call the Groq API
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    print(f'Screenshot saved to {path}')

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera is not open.')
        exit()

    # Capture frame from the webcam
    ret, frame = web_cam.read()

    # Check if frame was captured successfully
    if ret:
        path = 'webcam.jpg'
        cv2.imwrite(path, frame)
        print(f'Frame captured and saved to {path}')
    else:
        print('Error: Could not read frame from webcam.')

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

# Main loop
while True:
    prompt = input('USER:')
    if prompt.lower() == "exit":
        print("Exiting chat.")
        break  # Exits the loop if the user types 'exit'

    call = function_call(prompt)
    visual_context = None  # Default value in case no visual context is generated

    if 'take screenshot' in call:
        print('Taking screenshot')
        take_screenshot()
        visual_context = vision_prompt(prompt=prompt, photo_path='screenshot.jpg')
    elif 'capture webcam' in call:
        print('Capturing Webcam')
        web_cam_capture()
        visual_context = vision_prompt(prompt=prompt, photo_path='webcam.jpg')
    elif 'extract clipboard' in call:
        print('Copying Clipboard text')
        paste = get_clipboard_text()
        prompt = f'{prompt} \n\n CLIPBOARD CONTENT: {paste}'

    response = groq_prompt(prompt=prompt, img_context=visual_context)
    print(response)

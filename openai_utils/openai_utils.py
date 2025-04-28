import os
from typing import Optional, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
keywords = ["context","question"]
default_prompt_path = "prompts/default.txt"

class OpenaiAgent:

    def __init__(self, model: Optional[str] = None, prompt_path: Optional[str] = None, placeholders: Dict[str, str] = None):
        self.prompt_path = prompt_path
        self.placeholders = placeholders if placeholders else {}
        self.prompt = self.__get_prompt()
        self.model = model if model else "gpt-3.5-turbo"


    def __get_prompt(self):
        path = self.prompt_path if self.prompt_path and os.path.exists(self.prompt_path) else default_prompt_path
        with open(path, "r") as f:
            prompt = f.read()

        if self.placeholders and path != default_prompt_path:
            for keyword in keywords:
                prompt = prompt.replace('<{}>'.format(keyword), self.placeholders[keyword])

        return prompt

    def sync_call(self, content: str, prompt_path: Optional[str] = None, placeholders: Dict[str, str] = None):
        if prompt_path:
            self.prompt_path = prompt_path
            self.placeholders = placeholders
            if "question" not in placeholders:
                self.placeholders.update({"question": content})
            self.prompt = self.__get_prompt()

        messages = []
        if self.prompt:
            messages.append({"role": "user", "content": self.prompt})
        #messages.append({"role": "user", "content": content})

        response = client.chat.completions.create(
            model=self.model, messages=messages
        )

        return response.choices[0].message.content

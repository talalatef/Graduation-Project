import os
import requests

class LLAMAGenerator:
    def __init__(self, api_key, model="meta-llama/Meta-Llama-3-70B-Instruct"):
        self.api_base = "https://api.endpoints.anyscale.com/v1"
        self.url = f"{self.api_base}/chat/completions"
        self.token = api_key
        self.model = model

    def generate(self, context, query):
        prompt = (
    f"Imagine you're a renowned brain tumor specialist about to provide insights on a complex case. "
    f"Let's streamline the process for optimal efficiency. "
    f"If the question is related to brain tumors, continue with the process. If not, apologize to the user and inform them of your specific expertise in brain tumor medicine. "
    f"First, provide a summary answer to the Doctor's question. "
    f"Then, if the Doctor asks for a detailed answer, give a detailed and specific answer to help them in their work accurately. "
    f"Use the following context from the indexed database to generate your response: {context} "
    f"Question: {query} "
)



        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt}],
            "temperature": .7,
            "max_tokens": 1024,
            "top_p": 1,
            "frequency_penalty": 0
        }
        response = requests.post(self.url, headers={"Authorization": f"Bearer {self.token}"}, json=body)
        return response.json()["choices"][0]["message"]["content"]

# # Replace with your token
# llama_token = "esecret_c6dpbv7l3b4ba3d75n81cbv94z"
# generator = LLAMAGenerator(token=llama_token)

import http.client
import json
import time


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.n_trial = 5

    def get_response(self, prompt_content):
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
            }
        )

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": 1,
        }

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                conn = http.client.HTTPSConnection(self.api_endpoint, timeout=50)
                # conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
                res = conn.getresponse()
                data = res.read()
                json_data = json.loads(data)
                response = json_data["choices"][0]["message"]["content"]
                break
            except Exception as e:
                print(f"Error in API: {e}. Restarting the process...")
                time.sleep(2)
                continue
        return response

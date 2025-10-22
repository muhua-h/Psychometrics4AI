import os
from dotenv import load_dotenv
import requests
import json
from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
# from google import genai
# import anthropic
# from google.genai import types
# Load environment variables
load_dotenv()


def get_model_response(model, user_prompt,
                       system_prompt="You are a helpful assistant.",
                       temperature=1.0, max_tokens=2048):
    """
    Universal function to get responses from various AI models.

    Args:
        model (str): The model to use (e.g., "gpt-4", "claude", "llama", "deepseek", "gemini")
        user_prompt (str): The user's input prompt
        system_prompt (str): System instructions for the AI model
        temperature (float): Controls randomness (0.0 to 1.0)
        max_tokens (int): Maximum number of tokens to generate

    Returns:
        str: The model's response text
    """
    model = model.lower()

    # OpenAI models via Azure
    if model in ["gpt-4", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
                 "gpt-4.5-preview", "gpt-4o", "gpt-4o-mini"]:
        return _get_azure_openai_response(model, user_prompt, system_prompt,
                                          temperature, max_tokens)

    # DeepSeek via Azure AI Inference
    elif model == "deepseek":
        return _get_deepseek_response(user_prompt, system_prompt, temperature,
                                      max_tokens)

    # Llama via Fireworks.ai
    elif model == "llama":
        return _get_llama_response(user_prompt, system_prompt, temperature,
                                   max_tokens)

    # Claude via Anthropic
    elif model == "claude":
        return _get_claude_response(user_prompt, system_prompt, temperature,
                                    max_tokens)

    # Gemini via Google
    elif model == "gemini":
        return _get_gemini_response(user_prompt, system_prompt, temperature,
                                    max_tokens)

    else:
        raise ValueError(f"Unsupported model: {model}")


def _get_azure_openai_response(model_name, user_prompt, system_prompt,
                               temperature, max_tokens):
    """Get response from Azure-hosted OpenAI models or F5AI if key is set"""
    f5ai_key = os.getenv("F5AI_API_KEY")
    if f5ai_key and model_name in ["gpt-4o", "gpt-4"]:
        # Используем requests для F5AI (совместим с его форматом ответа)
        import requests
        url = "https://api.f5ai.ru/v2/chat/completions"
        headers = {
            "X-Auth-Token": f5ai_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            print(f"F5AI response status: {response.status_code}")
            print(f"F5AI response text: {response.text}")  # Для отладки полного ответа
            response.raise_for_status()  # Поднимает ошибку для 4xx/5xx
            json_response = response.json()
            # Парсинг по формату F5AI (из вашего curl)
            if "message" in json_response and "content" in json_response["message"]:
                return json_response["message"]["content"]
            else:
                raise KeyError(f"Unexpected format: {json_response}")
        except requests.exceptions.RequestException as e:
            print(f"F5AI request error: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"F5AI parse error: {e}")
            raise
    else:
        # Fallback на Azure (если ключ F5AI не задан)
        endpoint = "https://allmodelapi3225011299.openai.azure.com/"
        azure_key = os.getenv("AZURE_API_KEY")
        api_version = "2023-12-01-preview"

        deployment_map = {
            "gpt-4": "gpt-4",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
        }
        deployment = deployment_map.get(model_name, model_name)

        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=azure_key,
        )
        print(f"Using Azure for model: {model_name}")

        try:
            response = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure OpenAI error: {e}")
            raise

def _get_deepseek_response(user_prompt, system_prompt, temperature, max_tokens):
    """Get response from DeepSeek-V3 via F5AI (primary) or Azure fallback"""
    f5ai_key = os.getenv("F5AI_API_KEY")
    if f5ai_key:
        # Используем F5AI для DeepSeek (совместим с его форматом ответа)
        import requests
        url = "https://api.f5ai.ru/v2/chat/completions"
        headers = {
            "X-Auth-Token": f5ai_key,
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",  # Модель в F5AI (проверьте в /v2/models)
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            print(f"F5AI DeepSeek response status: {response.status_code}")
            print(f"F5AI DeepSeek response text: {response.text}")  # Для отладки
            response.raise_for_status()  # Поднимает ошибку для 4xx/5xx
            json_response = response.json()
            # Парсинг по формату F5AI
            if "message" in json_response and "content" in json_response["message"]:
                return json_response["message"]["content"]
            else:
                raise KeyError(f"Unexpected format: {json_response}")
        except requests.exceptions.RequestException as e:
            print(f"F5AI DeepSeek request error: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"F5AI DeepSeek parse error: {e}")
            raise
    else:
        # Fallback на Azure AI Inference (оригинальный код, если F5AI не задан)
        azure_key = os.getenv("AZURE_AI_INFERENCE_KEY")  # Правильный ключ для Inference
        endpoint = "https://allmodelapi3225011299.services.ai.azure.com/models"  # Ваш Azure endpoint
        model_name = "DeepSeek-V3"

        try:
            client = ChatCompletionsClient(
                endpoint=endpoint,
                credential=AzureKeyCredential(azure_key),
            )

            response = client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                model=model_name
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek Azure error: {e}")
            raise

def _get_llama_response(user_prompt, system_prompt, temperature, max_tokens):
    """ Get response from llama model via Azure AI Inference"""
    azure_key = os.getenv("AZURE_API_KEY")
    endpoint = "https://allmodelapi3225011299.services.ai.azure.com/models"
    model_name = "Llama-3.3-70B-Instruct"

    try:
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(azure_key),

        )

        response = client.complete(
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=user_prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            model=model_name
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Llama error: {e}")
        raise






def _get_claude_response(user_prompt, system_prompt, temperature, max_tokens):
    """Get response from Anthropic's Claude model"""
    api_key=os.environ.get("CLAUDE_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing ANTHROPIC_API_KEY or CLAUDE_API_KEY in environment variables")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Claude API error: {e}")
        raise


def _get_gemini_response(user_prompt, system_prompt, temperature=0.7, max_tokens=512):
    """Get response from Gemini using generate_content (single-turn)"""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    try:
        client = genai.Client(api_key=api_key)

        # Combine system and user prompts into one Content object
        content = types.Content(
            role="user",
            parts=[
                types.Part(text=f"{system_prompt}\n\n{user_prompt}")
            ]
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[content],
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.text

    except Exception as e:
        print(f"Gemini API error: {e}")
        raise
if __name__ == "__main__":

    ## Test all models

    # deepseek
    response = get_model_response(
        model="deepseek",
        user_prompt="What are three benefits of AI models in modern applications?",
        system_prompt="You are a technical expert who explains concepts clearly and concisely."
    )
    print(f"deepseek response:\n{response}\n")

    # # llama
    # response = get_model_response(
    #     model="llama",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )

    # print(f"llama response:\n{response}\n")

    # # claude
    # response = get_model_response(
    #     model="claude",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )
    # print(f"claude response:\n{response}\n")

    # # gemini
    # response = get_model_response(
    #     model="gemini",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )
    # print(f"gemini response:\n{response}\n")

    # # gpt-4
    # response = get_model_response(
    #     model="gpt-4",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )
    # print(f"gpt-4 response:\n{response}\n")
    #
    # # gpt-4o
    # response = get_model_response(
    #     model="gpt-4o",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )
    # print(f"gpt-4o response:\n{response}\n")
    #
    # # gpt-4.5-preview
    # response = get_model_response(
    #     model="gpt-4.5-preview",
    #     user_prompt="What are three benefits of AI models in modern applications?",
    #     system_prompt="You are a technical expert who explains concepts clearly and concisely."
    # )
    # print(f"gpt-4.5-preview response:\n{response}\n")

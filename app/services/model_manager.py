import os
from typing import Any, Dict, Optional

class ModelManager:
    """
    Manages the loading and inference of local models stored in the 'models/' directory.
    Designed to support Hugging Face models, GGUF, or other formats.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.loaded_models: Dict[str, Any] = {}
        # Environment variable for the optimized host we just built
        self.llm_host_url = os.getenv("LLM_HOST_URL", "http://llm-host:8000")

    def list_available_models(self) -> list[str]:
        """Lists all model files in the models directory."""
        if not os.path.exists(self.models_dir):
            return ["optimized-code-agent"] # Always advertise our hosted model
        local_files = [f for f in os.listdir(self.models_dir) if not f.startswith('.')]
        return ["optimized-code-agent"] + local_files

    def load_model(self, model_name: str, model_type: str = "dummy"):
        """
        Loads a model into memory.
        For the optimized host, this is a no-op or a health check.
        """
        if model_name == "optimized-code-agent":
            print(f"Model {model_name} is managed by external host at {self.llm_host_url}")
            self.loaded_models[model_name] = RemoteModelClient(self.llm_host_url)
            return self.loaded_models[model_name]

        model_path = os.path.join(self.models_dir, model_name)
        
        if model_name in self.loaded_models:
            print(f"Model {model_name} is already loaded.")
            return self.loaded_models[model_name]
            
        print(f"Loading model {model_name} from {model_path}...")
        
        try:
            if model_type == "dummy":
                self.loaded_models[model_name] = DummyModel(model_name)
            elif model_type == "hf":
                print(f"Initializing HF model from {model_path}")
                self.loaded_models[model_name] = DummyModel(model_name) 
            elif model_type == "llama_cpp":
                print(f"Initializing LlamaCPP model from {model_path}")
                self.loaded_models[model_name] = DummyModel(model_name)
            
            print(f"Successfully loaded {model_name}.")
            return self.loaded_models[model_name]
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise e

    async def predict(self, model_name: str, prompt: str, **kwargs) -> str:
        """Runs inference on the specified model."""
        if model_name not in self.loaded_models:
            # Auto-load logic
            if model_name == "optimized-code-agent":
                self.load_model(model_name)
            else:
                self.load_model(model_name)
            
        model = self.loaded_models[model_name]
        
        if isinstance(model, RemoteModelClient):
            return await model.generate(prompt, **kwargs)
        else:
            return model.generate(prompt, **kwargs)  # Synchronous dummy for legacy

class RemoteModelClient:
    """Client to talk to the optimized LLM Host service."""
    def __init__(self, host_url: str):
        self.host_url = host_url
        
    async def generate(self, prompt: str, **kwargs) -> str:
        import aiohttp
        url = f"{self.host_url}/generate"
        payload = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_new_tokens", 256),
            "temperature": kwargs.get("temperature", 0.2),
            "do_sample": kwargs.get("do_sample", True)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("generated_text", "")
                    else:
                        return f"Error from Host: {response.status}"
        except Exception as e:
            return f"Failed to contact LLM Host: {e}"

class DummyModel:
    """A mock model class that simulates inference."""
    def __init__(self, name: str):
        self.name = name
        
    def generate(self, prompt: str, **kwargs) -> str:
        return f"[Local Model {self.name}]: Processed prompt '{prompt[:20]}...' -> This is a simulated response."

# Singleton instance
model_manager = ModelManager()

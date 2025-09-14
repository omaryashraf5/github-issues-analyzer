#!/usr/bin/env python3
"""
Test script to check Ollama server connectivity and embeddings
"""

import json
import sys
import time

import requests


def test_ollama_connection(base_url="http://localhost:11434"):
    """Test basic connection to Ollama server"""
    print(f"🔍 Testing connection to Ollama server at {base_url}")

    try:
        # Test basic tags endpoint
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
            models = response.json()
            available_models = [model["name"] for model in models.get("models", [])]
            print(f"Available models: {available_models}")
            return True, available_models
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is Ollama running?")
        print("Try starting it with: ollama serve")
        return False, []
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False, []


def test_embeddings_endpoint(
    base_url="http://localhost:11434", model_name="nomic-embed-text"
):
    """Test embeddings endpoint"""
    print(f"🧠 Testing embeddings endpoint with model: {model_name}")

    try:
        # Test with a simple text
        test_text = "This is a test issue about API connectivity"

        request_data = {"model": model_name, "prompt": test_text}

        print("📤 Sending test embedding request...")

        response = requests.post(
            f"{base_url}/api/embeddings",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            if "embedding" in result and len(result["embedding"]) > 0:
                embedding = result["embedding"]
                print(f"✅ Embeddings endpoint working!")
                print(f"   Embedding dimension: {len(embedding)}")
                print(f"   Sample values: {embedding[:5]}...")
                return True, len(embedding)
            else:
                print("❌ No embedding in response")
                print(f"Response: {result}")
                return False, 0
        else:
            print(f"❌ Embeddings request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False, 0

    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False, 0


def pull_model(base_url="http://localhost:11434", model_name="nomic-embed-text"):
    """Pull a model if it's not available"""
    print(f"📥 Attempting to pull model: {model_name}")

    try:
        response = requests.post(
            f"{base_url}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=300,  # 5 minutes timeout
        )

        if response.status_code == 200:
            print("Pulling model... This may take a few minutes.")
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        data = json.loads(chunk.decode())
                        if "status" in data:
                            print(f"📥 {data['status']}")
                        if data.get("status") == "success":
                            print(f"✅ Successfully pulled {model_name}")
                            return True
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"❌ Failed to pull model: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False


def main():
    """Run all tests"""
    print("🦙 Ollama Server Test Suite")
    print("=" * 50)

    # Get server URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "nomic-embed-text"

    print(f"Server URL: {base_url}")
    print(f"Model: {model_name}")
    print()

    # Test 1: Basic connection
    success, available_models = test_ollama_connection(base_url)
    if not success:
        print("\n🚨 Basic connection failed. Make sure Ollama is running.")
        print("Try starting it with: ollama serve")
        sys.exit(1)

    print()

    # Test 2: Check if model is available
    if model_name not in available_models:
        print(f"⚠️  Model '{model_name}' not found in available models")
        print("Attempting to pull the model...")

        if pull_model(base_url, model_name):
            print("✅ Model successfully pulled")
        else:
            print("❌ Failed to pull model")
            print("You can manually pull it with:")
            print(f"   ollama pull {model_name}")
            sys.exit(1)
    else:
        print(f"✅ Model '{model_name}' is available")

    print()

    # Test 3: Test embeddings
    success, dimension = test_embeddings_endpoint(base_url, model_name)

    print()
    print("=" * 50)

    if success:
        print("🎉 All tests passed! Ollama server is ready for use.")
        print(f"✅ Server: {base_url}")
        print(f"✅ Model: {model_name}")
        print(f"✅ Embedding dimension: {dimension}")
        print()
        print("You can now run the analyzer with:")
        print(
            f"python main.py --ollama --ollama-url {base_url} --ollama-model {model_name}"
        )
    else:
        print("❌ Embeddings test failed. Check your setup.")
        print()
        print("Common issues:")
        print("1. Model not loaded or available")
        print("2. Wrong model name")
        print("3. Server not configured for embeddings")
        print("4. Model doesn't support embeddings")
        print()
        print("Popular embedding models for Ollama:")
        print("- nomic-embed-text (recommended)")
        print("- all-minilm")
        print("- mxbai-embed-large")


if __name__ == "__main__":
    main()

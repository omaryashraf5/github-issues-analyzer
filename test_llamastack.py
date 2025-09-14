#!/usr/bin/env python3
"""
Test script to check Llama Stack server connectivity and embeddings
"""

import json
import sys
import time

import requests


def test_llamastack_connection(base_url="http://localhost:8321"):
    """Test basic connection to Llama Stack server"""
    print(f"🔍 Testing connection to Llama Stack server at {base_url}")

    try:
        # Test basic health endpoint (correct Llama Stack endpoint)
        response = requests.get(f"{base_url}/v1/health", timeout=10)
        if response.status_code == 200:
            print("✅ Server is responding")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_embeddings_endpoint(
    base_url="http://localhost:8321", model_id="sentence-transformers/all-MiniLM-L6-v2"
):
    """Test embeddings endpoint"""
    print(f"🧠 Testing embeddings endpoint with model: {model_id}")

    try:
        # Test with a simple text
        test_text = ["This is a test issue about API connectivity"]

        request_data = {"model_id": model_id, "contents": test_text}

        print("📤 Sending test embedding request...")

        response = requests.post(
            f"{base_url}/v1/inference/embeddings",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            if "embeddings" in result and len(result["embeddings"]) > 0:
                embedding = result["embeddings"][0]
                print(f"✅ Embeddings endpoint working!")
                print(f"   Embedding dimension: {len(embedding)}")
                print(f"   Sample values: {embedding[:5]}...")
                return True, len(embedding)
            else:
                print("❌ No embeddings in response")
                print(f"Response: {result}")
                return False, 0
        else:
            print(f"❌ Embeddings request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False, 0

    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False, 0


def list_available_models(base_url="http://localhost:8321"):
    """Try to list available models"""
    print("📋 Attempting to list available models...")

    # Try different possible endpoints
    endpoints_to_try = [
        "/models",
        "/v1/openai/v1/models",
        "/api/models",
        "/inference/models",
    ]

    for endpoint in endpoints_to_try:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Found models endpoint: {endpoint}")
                print(f"Available models: {result}")
                return result
        except:
            continue

    print("⚠️  Could not find models endpoint")
    return None


def main():
    """Run all tests"""
    print("🦙 Llama Stack Server Test Suite")
    print("=" * 50)

    # Get server URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8321"
    model_id = (
        sys.argv[2] if len(sys.argv) > 2 else "sentence-transformers/all-MiniLM-L6-v2"
    )

    print(f"Server URL: {base_url}")
    print(f"Model ID: {model_id}")
    print()

    # Test 1: Basic connection
    if not test_llamastack_connection(base_url):
        print("\n🚨 Basic connection failed. Make sure the server is running.")
        print("Try starting it with: llama stack run <your-config>")
        sys.exit(1)

    print()

    # Test 2: List models (optional)
    list_available_models(base_url)
    print()

    # Test 3: Test embeddings
    success, dimension = test_embeddings_endpoint(base_url, model_id)

    print()
    print("=" * 50)

    if success:
        print("🎉 All tests passed! Llama Stack server is ready for use.")
        print(f"✅ Server: {base_url}")
        print(f"✅ Model: {model_id}")
        print(f"✅ Embedding dimension: {dimension}")
        print()
        print("You can now run the analyzer with:")
        print(
            f"python main.py --llamastack --llamastack-url {base_url} --llamastack-model {model_id}"
        )
    else:
        print("❌ Embeddings test failed. Check your server configuration.")
        print()
        print("Common issues:")
        print("1. Model not loaded or available")
        print("2. Wrong model ID")
        print("3. Server not configured for embeddings")
        print("4. API endpoint mismatch")


if __name__ == "__main__":
    main()

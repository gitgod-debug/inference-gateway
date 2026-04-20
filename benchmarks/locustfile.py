"""Locust load test for the Inference Gateway.

Usage:
  locust -f benchmarks/locustfile.py --host=http://localhost:8080 --headless -u 50 -r 5 -t 60s
"""

from locust import HttpUser, between, task


class GatewayUser(HttpUser):
    """Simulates a typical gateway user."""

    wait_time = between(0.5, 2.0)
    headers = {
        "Authorization": "Bearer ig-demo-key-2026",
        "Content-Type": "application/json",
    }

    @task(10)
    def chat_completion(self):
        """Non-streaming chat completion."""
        self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
        )

    @task(3)
    def chat_streaming(self):
        """Streaming chat completion."""
        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Tell me a joke"}],
                "stream": True,
                "max_tokens": 100,
            },
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                for _ in resp.iter_lines():
                    pass
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def list_models(self):
        """List available models."""
        self.client.get("/v1/models", headers=self.headers)

    @task(1)
    def health_check(self):
        """Health check."""
        self.client.get("/health")

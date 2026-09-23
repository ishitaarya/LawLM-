from fastapi.testclient import TestClient

from app import app


def main():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "LawLM"
    assert data["status"] == "running"

    print("=" * 60)
    print("LawLM - API Test")
    print("=" * 60)
    print("Root endpoint test passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

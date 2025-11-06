import sys, requests

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_request.py <image_path> [api_base]")
        return
    image_path = sys.argv[1]
    api_base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"

    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        r = requests.post(f"{api_base}/predict", files=files)
    print("Status:", r.status_code)
    try:
        print("JSON:", r.json())
    except Exception:
        print("Text:", r.text)

if __name__ == "__main__":
    main()
